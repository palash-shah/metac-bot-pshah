import argparse
import asyncio
import logging
from datetime import datetime, timezone
import dotenv
from typing import Literal
import threading
import time
from statistics import mean, pstdev


from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

import os
from google import genai
from openai import OpenAI


dotenv.load_dotenv()
logger = logging.getLogger(__name__)

_OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class GeminiLlm:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0, timeout: int = 30):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    async def invoke(self, prompt: str) -> str:
        # client is sync; run in thread to avoid blocking
        resp = await asyncio.to_thread(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": self.temperature},
            )
        )
        text = resp.text
        # response text location may be resp.candidates[0].content
        return resp.candidates[0].content

class RateLimitedMetaculusClient:
    """
    Wraps a MetaculusClient instance to ensure:
    - only one call runs at a time (single connection)
    - at least min_interval seconds between calls
    All methods from the wrapped client are proxied synchronously.
    """
    def __init__(self, client, min_interval: float = 10.0):
        self._client = client
        self._lock = threading.Lock()
        self._min_interval = float(min_interval)
        self._last_call = 0.0

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            with self._lock:
                now = time.time()
                wait = self._min_interval - (now - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                result = attr(*args, **kwargs)
                self._last_call = time.time()
                return result

        return _wrapped

class SpringTemplateBot2026(ForecastBot):
    """
    This is the template bot for Spring 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Spring AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """
    _structure_output_validation_samples = 2
    _asknews_min_interval = 11.0
    _last_asknews_call = 0.0
    _asknews_lock = None


    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # regime_state can track last regime for each question id
        self._regime_state: dict[int, int] = {}
        # P[a][b] = transition prob; load from file or hard‑code for now
        self._regime_transition: dict[int, dict[int, float]] = {}
        # optional: regime‑specific “anchor” forecasts
        self._regime_anchors: dict[int, dict[str, float]] = {}
        # do NOT touch client, rate limiter, etc.
                # Example: 2 regimes, calm (0) and noisy (1)
        self._regime_transition = {
            0: {0: 0.9, 1: 0.1},
            1: {0: 0.3, 1: 0.7},
        }

        # These anchors should be filled from your offline clustering
        # (placeholder values shown)
        self._regime_anchors[0] = {
            "binary_default": 0.02,          # calm regime: near crowd 2%
            "numeric_scale": 1.0,
            "numeric_shift": 0.0,
        }
        self._regime_anchors[1] = {
            "binary_default": 0.10,          # noisy regime: more uncertainty
            "numeric_scale": 1.2,            # widen numeric distributions
            "numeric_shift": 0.0,
        }


    ##################### REGIME MODEL HELPERS #######################
    def _get_recent_community_series( ### NEED TO ADJUST TO MATCH API JSON AND CHANGE THE API OF METACULUSQUESTION CUSTOM METADATA FIELD TO COMMUNITY VOLATILITY SENTIMENT ###
        self,
        question: MetaculusQuestion,
        max_points: int = 30,
    ) -> list[float]:
        """
        Return up to max_points most recent community probabilities in [0,1]
        for this question, using raw api_json.
        """
        q_json = question.api_json.get("question", {})
        # Adjust this key path to match what you see in actual API JSON.
        # Common patterns are something like "community_prediction", "cp_history", etc.
        history = q_json.get("community_prediction_history") or []
        if not history:
            return []

        # history items should have timestamp + value; sort and take last N
        # e.g. [{"t": 1700000000, "cp": 0.12}, ...]
        history = sorted(history, key=lambda h: h["t"])[-max_points:]
        path = [float(h["cp"]) for h in history]
        return path
    
    def _get_current_regime(self, question: MetaculusQuestion) -> int:
        """
        Map question + community history into a discrete regime index.
        This is where you call your spectral clustering classifier that
        was trained offline on community prediction time series.
        """
        qid = question.page_url  # or question.page_url
        # If you have a saved classifier:
        # features = build_features_from_question(question)
        # regime = self.spectral_classifier.predict(features)
        # For now, fall back to previous regime or default:
        return self._regime_state.get(qid, 0)

    def _forecast_next_regime_probs(self, current_regime: int) -> dict[int, float]:
        """
        Use empirical transition matrix P (from the paper) to get
        P(R_{t+1} = k | R_t).
        """
        row = self._regime_transition.get(current_regime)
        if not row:
            # uniform fallback
            return {k: 1.0 for k in (0,)}
        return row

    def _regime_weighted_binary_anchor(self, regime_probs: dict[int, float]) -> float:
        """
        Turn regime probabilities into an anchor probability in [0,1].
        Example: each regime k has an anchor p_k; return sum_k pi_k p_k.
        """
        total = 0.0
        norm = 0.0
        for k, pi_k in regime_probs.items():
            anchors = self._regime_anchors.get(k, {})
            p_k = anchors.get("binary_default", 0.5)
            total += pi_k * p_k
            norm += pi_k
        return total / norm if norm > 0 else 0.5


    ################ WAIT BETWEEN QUESTION CALLS FOR API APPLIANCE #######################

    async def _wait_for_asknews(self) -> None:
        # Lazily create an asyncio.Lock so we don't create one at import time
        if self._asknews_lock is None:
            self._asknews_lock = asyncio.Lock()
        async with self._asknews_lock:
            now = time.time()
            elapsed = now - self._last_asknews_call
            wait = self._asknews_min_interval - elapsed
            if wait > 0:
                logger.info(f"Waiting {wait:.1f}s to respect AskNews rate limit")
                await asyncio.sleep(wait)
            # Record this attempt time so the next call is spaced
            self._last_asknews_call = time.time()

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                searcher = AskNewsSearcher()
                max_tries = 3
                research = ""
                for attempt in range(1, max_tries + 1):
                    await self._wait_for_asknews()
                    try:
                        research = await searcher.call_preconfigured_version(
                            researcher, prompt
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"AskNews call failed (attempt {attempt}/{max_tries}): {e}"
                        )
                        if attempt < max_tries:
                            await asyncio.sleep(2 ** attempt)  # exponential backoff
                        else:
                            research = f"[Error fetching research after {max_tries} attempts: {e}]"
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self,
        question: BinaryQuestion,
        research: str,
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        # Original LLM-based forecast
        base_prediction: ReasonedPrediction[float] = await self._binary_prompt_to_forecast(
            question,
            prompt,
        )

        # -------- Regime ensemble adjustment (logic only) --------
        # These helpers should be defined on the class (no API / wait logic touched).

        current_regime: int = self._get_current_regime(question)
        next_regime_probs: dict[int, float] = self._forecast_next_regime_probs(current_regime)
        anchor_prob: float = self._regime_weighted_binary_anchor(next_regime_probs)

        llm_p: float = float(base_prediction.prediction_value)
        alpha: float = 0.3  # weight on regime prior; tune offline

        blended_p: float = alpha * anchor_prob + (1.0 - alpha) * llm_p
        blended_p = max(0.01, min(0.99, blended_p))

        # update internal regime state for this question id
        if next_regime_probs:
            self._regime_state[question.page_url] = max(
                next_regime_probs,
                key=next_regime_probs.get,
            )
        else:
            self._regime_state[question.page_url] = current_regime

        reasoning = (
            base_prediction.reasoning
            + f"\n\n[Regime ensemble] Current regime {current_regime}, "
            f"anchor={anchor_prob:.3f}, LLM={llm_p:.3f}, blended={blended_p:.3f}."
        )

        return ReasonedPrediction(
            prediction_value=blended_p,
            reasoning=reasoning,
        )

    async def _binary_prompt_to_forecast(
            self,
            question: BinaryQuestion,
            prompt: str,
        ) -> ReasonedPrediction[float]:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
            binary_prediction: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

            logger.info(
                f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
            )
            return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self,
        question: MultipleChoiceQuestion,
        research: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are:
            {question.options}

            Background:
            {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            You write your rationale remembering that:
            1) Good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and
            2) Good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order:
            {question.options}
            as:
            OptionA: ProbabilityA
            OptionB: ProbabilityB
            ...
            OptionN: ProbabilityN
            """
        )

        base_prediction: ReasonedPrediction[PredictedOptionList] = (
            await self._multiple_choice_prompt_to_forecast(question, prompt)
        )

        # -------- Regime ensemble adjustment (logic only) --------
        current_regime: int = self._get_current_regime(question)
        next_regime_probs: dict[int, float] = self._forecast_next_regime_probs(current_regime)

        predicted_option_list = base_prediction.prediction_value
        print("PREDICTED_OPTION_LIST:", predicted_option_list)
        print("PREDICTED_OPTION_LIST_DICT:", predicted_option_list.__dict__)
        options = predicted_option_list.predicted_options  # <- field name from forecastingtools

        p_llm: dict[str, float] = {
            opt.option_name: float(opt.probability)
            for opt in options
        }


        # Build regime-mixture prior over options
        prior: dict[str, float] = {opt.option_name: 0.0 for opt in options}
        for k, pi_k in next_regime_probs.items():
            anchors = self._regime_anchors.get(k, {})
            for opt in options:
                key = f"mc_{opt.option_name}"
                prior_k = float(anchors.get(key, 1.0 / len(options)))
                prior[opt.option_name] += pi_k * prior_k

        alpha: float = 0.3  # weight on regime prior
        blended: dict[str, float] = {}
        for name in prior:
            blended[name] = alpha * prior[name] + (1.0 - alpha) * p_llm[name]

        # Renormalize to sum to 1
        total_prob = sum(blended.values()) or 1.0
        for opt in options:
            opt.probability = blended[opt.option_name] / total_prob

        if next_regime_probs:
            self._regime_state[question.page_url] = max(
                next_regime_probs,
                key=next_regime_probs.get,
            )
        else:
            self._regime_state[question.page_url] = current_regime

        reasoning = (
            base_prediction.reasoning
            + "\n\n[Regime ensemble] Blended LLM option probabilities "
            "with regime-mixture prior over options."
        )

        return ReasonedPrediction(
            prediction_value=base_prediction.prediction_value,
            reasoning=reasoning,
        )

    async def _multiple_choice_prompt_to_forecast(
            self,
            question: MultipleChoiceQuestion,
            prompt: str,
        ) -> ReasonedPrediction[PredictedOptionList]:
            parsing_instructions = clean_indents(
                f"""
                Make sure that all option names are one of the following:
                {question.options}

                The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
                Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
                """
            )
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
            predicted_option_list: PredictedOptionList = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=parsing_instructions,
            )

            logger.info(
                f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
            )
            return ReasonedPrediction(
                prediction_value=predicted_option_list, reasoning=reasoning
            )

        ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self,
        question: NumericQuestion,
        research: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(
            question
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated, please infer this."}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.
            - The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            You remind yourself that good forecasters are humble and set wide 90–10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            """
        )

        base_forecast = await self._numeric_prompt_to_forecast(question, prompt)

        # -------- Regime ensemble adjustment (logic only) --------
        current_regime: int = self._get_current_regime(question)
        next_regime_probs: dict[int, float] = self._forecast_next_regime_probs(current_regime)

        # Aggregate regime-specific scale / shift anchors
        scale: float = 1.0
        shift: float = 0.0
        for k, pi_k in next_regime_probs.items():
            anchors = self._regime_anchors.get(k, {})
            scale_k = float(anchors.get("numeric_scale", 1.0))
            shift_k = float(anchors.get("numeric_shift", 0.0))
            scale += pi_k * (scale_k - 1.0)
            shift += pi_k * shift_k

        dist = base_forecast.prediction_value
        new_percentiles: list[Percentile] = []

        for p in dist.declared_percentiles:
            v = float(p.value)
            adj_v = v * scale + shift
            new_percentiles.append(
                Percentile(
                    percentile=p.percentile,
                    value=adj_v,
                )
            )

        adjusted_dist = NumericDistribution.from_question(
            new_percentiles,
            question,
        )

        if next_regime_probs:
            self._regime_state[question.page_url] = max(
                next_regime_probs,
                key=next_regime_probs.get,
            )
        else:
            self._regime_state[question.page_url] = current_regime

        reasoning = (
            base_forecast.reasoning
            + f"\n\n[Regime ensemble] Applied scale={scale:.3f}, shift={shift:.3f} "
            f"based on volatility regime."
        )

        return ReasonedPrediction(
            prediction_value=adjusted_dist,
            reasoning=reasoning,
        )

    async def _numeric_prompt_to_forecast(
            self,
            question: NumericQuestion,
            prompt: str,
        ) -> ReasonedPrediction[NumericDistribution]:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
            parsing_instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a numeric question.
                - This text is trying to answer the numeric question: "{question.question_text}".
                - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
                - The units for the forecast are: {question.unit_of_measure}
                - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
                - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
                - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                - Turn any values that are in scientific notation into regular numbers.
                """
            )
            percentile_list: list[Percentile] = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions,
                num_validation_samples=self._structure_output_validation_samples,
            )
            prediction = NumericDistribution.from_question(percentile_list, question)
            logger.info(
                f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
            )
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

        ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
            self, question: DateQuestion, research: str
        ) -> ReasonedPrediction[NumericDistribution]:
            upper_bound_message, lower_bound_message = (
                self._create_upper_and_lower_bound_messages(question)
            )
            prompt = clean_indents(
                f"""
                You are a professional forecaster interviewing for a job.

                Your interview question is:
                {question.question_text}

                Background:
                {question.background_info}

                {question.resolution_criteria}

                {question.fine_print}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                {lower_bound_message}
                {upper_bound_message}

                Formatting Instructions:
                - This is a date question, and as such, the answer must be expressed in terms of dates.
                - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
                - Always start with a lower date chronologically and then increase from there.
                - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The outcome if nothing changed.
                (c) The outcome if the current trend continued.
                (d) The expectations of experts and markets.
                (e) A brief description of an unexpected scenario that results in a low outcome.
                (f) A brief description of an unexpected scenario that results in a high outcome.

                {self._get_conditional_disclaimer_if_necessary(question)}
                You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

                The last thing you write is your final answer as:
                "
                Percentile 10: YYYY-MM-DD (oldest date)
                Percentile 20: YYYY-MM-DD
                Percentile 40: YYYY-MM-DD
                Percentile 60: YYYY-MM-DD
                Percentile 80: YYYY-MM-DD
                Percentile 90: YYYY-MM-DD (newest date)
                "
                """
            )
            forecast = await self._date_prompt_to_forecast(question, prompt)
            return forecast

    async def _date_prompt_to_forecast(
            self,
            question: DateQuestion,
            prompt: str,
        ) -> ReasonedPrediction[NumericDistribution]:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
            parsing_instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a date question.
                - This text is trying to answer the question: "{question.question_text}".
                - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
                - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                """
            )
            date_percentile_list: list[DatePercentile] = await structure_output(
                reasoning,
                list[DatePercentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions,
                num_validation_samples=self._structure_output_validation_samples,
            )

            percentile_list = [
                Percentile(
                    percentile=percentile.percentile,
                    value=percentile.value.timestamp(),
                )
                for percentile in date_percentile_list
            ]
            prediction = NumericDistribution.from_question(percentile_list, question)
            logger.info(
                f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
            )
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
            self, question: NumericQuestion | DateQuestion
        ) -> tuple[str, str]:
            if isinstance(question, NumericQuestion):
                if question.nominal_upper_bound is not None:
                    upper_bound_number = question.nominal_upper_bound
                else:
                    upper_bound_number = question.upper_bound
                if question.nominal_lower_bound is not None:
                    lower_bound_number = question.nominal_lower_bound
                else:
                    lower_bound_number = question.lower_bound
                unit_of_measure = question.unit_of_measure
            elif isinstance(question, DateQuestion):
                upper_bound_number = question.upper_bound.date().isoformat()
                lower_bound_number = question.lower_bound.date().isoformat()
                unit_of_measure = ""
            else:
                raise ValueError()

            if question.open_upper_bound:
                upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
            else:
                upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

            if question.open_lower_bound:
                lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
            else:
                lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
            return upper_bound_message, lower_bound_message

        ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
            self, question: ConditionalQuestion, research: str
        ) -> ReasonedPrediction[ConditionalPrediction]:
            parent_info, full_research = await self._get_question_prediction_info(
                question.parent, research, "parent"
            )
            child_info, full_research = await self._get_question_prediction_info(
                question.child, research, "child"
            )
            yes_info, full_research = await self._get_question_prediction_info(
                question.question_yes, full_research, "yes"
            )
            no_info, full_research = await self._get_question_prediction_info(
                question.question_no, full_research, "no"
            )
            full_reasoning = clean_indents(
                f"""
                ## Parent Question Reasoning
                {parent_info.reasoning}
                ## Child Question Reasoning
                {child_info.reasoning}
                ## Yes Question Reasoning
                {yes_info.reasoning}
                ## No Question Reasoning
                {no_info.reasoning}
            """
            )
            full_prediction = ConditionalPrediction(
                parent=parent_info.prediction_value,  # type: ignore
                child=child_info.prediction_value,  # type: ignore
                prediction_yes=yes_info.prediction_value,  # type: ignore
                prediction_no=no_info.prediction_value,  # type: ignore
            )
            return ReasonedPrediction(
                reasoning=full_reasoning, prediction_value=full_prediction
            )

    async def _get_question_prediction_info(
            self, question: MetaculusQuestion, research: str, question_type: str
        ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
            from forecasting_tools.data_models.data_organizer import DataOrganizer

            previous_forecasts = question.previous_forecasts
            if (
                question_type in ["parent", "child"]
                and previous_forecasts
                and question_type not in self.force_reforecast_in_conditional
            ):
                # TODO: add option to not affirm current parent/child forecasts, create new forecast
                previous_forecast = previous_forecasts[-1]
                current_utc_time = datetime.now(timezone.utc)
                if (
                    previous_forecast.timestamp_end is None
                    or previous_forecast.timestamp_end > current_utc_time
                ):
                    pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                    prediction = ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                    )
                    return (prediction, research)  # type: ignore
            info = await self._make_prediction(question, research)
            full_research = self._add_reasoning_to_research(research, info, question_type)
            return info, full_research  # type: ignore

    def _add_reasoning_to_research(
            self,
            research: str,
            reasoning: ReasonedPrediction[PredictionTypes],
            question_type: str,
        ) -> str:
            from forecasting_tools.data_models.data_organizer import DataOrganizer

            question_type = question_type.title()
            return clean_indents(
                f"""
                {research}
                ---
                ## {question_type} Question Information
                You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
                This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
                The reasoning for the {question_type} Question was as such:
                ```
                {reasoning.reasoning}
                ```
                This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
                """
            )

    def _get_conditional_disclaimer_if_necessary(
            self, question: MetaculusQuestion
        ) -> str:
            if question.conditional_type not in ["yes", "no"]:
                return ""
            return clean_indents(
                """
                As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
                You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
                """
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openai/gpt-4o-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )

    client = RateLimitedMetaculusClient(MetaculusClient(), min_interval=10.0)
    if run_mode == "tournament":
        # You may want to change this to the specific tournament ID you want to forecast on
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
