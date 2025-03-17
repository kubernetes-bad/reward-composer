import asyncio
import json
import logging
import re
import statistics
import threading
import time
from asyncio import Semaphore
from itertools import cycle
from typing import List, Optional, Callable, Sequence, Union
import random
import aiohttp
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import jinja2

from .rewards import RewardFunction, timed_execution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMReward")


# for default score parser. You're encouraged to write your own parser
score_float_pattern_str = r"(?:score|rating):\s*(?:\[)?(\d+(?:\.\d+)?)(?:\s*\/\s*(\d+(?:\.\d+)?))?(?:\])?"
score_pattern = re.compile(score_float_pattern_str, re.IGNORECASE)


class APIKeyManager:
    """
    Manages a pool of API keys with round-robin rotation and rate limit tracking.
    """

    def __init__(self, api_keys: Sequence[str]):
        """
        Initialize the API key manager.

        Args:
            :param api_keys: List or tuple of API keys to use
        """
        if not api_keys:
            raise ValueError("At least one API key must be provided")

        self.api_keys = list(api_keys)
        self.key_cycle = cycle(self.api_keys)
        self.current_key_index = 0
        self.rate_limited_keys = {} # maps key to timestamp when it can be used again
        self.lock = threading.RLock()

    def get_next_key(self) -> str:
        """
        Get the next available API key using round-robin rotation.
        If a key is rate limited, it will be skipped.

        Returns:
            :return The next available API key
        """
        with self.lock:
            now = time.time()

            for _ in range(len(self.api_keys)):
                key = next(self.key_cycle)

                if key not in self.rate_limited_keys:
                    return key

                resume_time = self.rate_limited_keys[key]
                if now < resume_time:
                    continue # still rate limited, try next one
                else:
                    del self.rate_limited_keys[key] # key is available!

            # if we got to here - all keys are rate limited
            # find the key that will become available soonest and wait for it
            if self.rate_limited_keys:
                key, resume_time = min(self.rate_limited_keys.items(), key=lambda x: x[1])
                wait_time = resume_time - now
                logger.warning(f"All API keys are rate limited. Waiting {wait_time:.2f} seconds for the next available key.")
                time.sleep(wait_time)
                del self.rate_limited_keys[key]
                return key
            else:
                # how tf did you get here
                logger.warning("No API keys available but none are rate limited. What is even going on. Using the first key.")
                return self.api_keys[0]

    def mark_rate_limited(self, key: str, retry_after: float = 60.0) -> None:
        """
        Mark an API key as rate limited.

        Args:
            :param key: The API key that was rate limited
            :param retry_after: Number of seconds to wait before using this key again
        """
        with self.lock:
            self.rate_limited_keys[key] = time.time() + retry_after
            logger.warning(f"API key marked as rate limited. Will become available after {retry_after} seconds.")


class LLMReward(RewardFunction):
    """
    Reward function that uses an LLM to evaluate completions.

    This reward function sends each completion to an OpenAI-compatible API
    and parses the response to extract a score.
    """

    def __init__(
        self,
        api_url: str,
        api_key: Union[str, Sequence[str]],
        prompt_template: str,
        model: str = "Oniichat/Onii-1.3-13b", # come on, set your model
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        output_parser: Optional[Callable[[str], float]] = None, # please use one!
        max_concurrent_requests: int = 10,
        max_retries: int = 10,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        ensemble_size: int = 1,
        ensemble_aggregation: str = "median",  # "mean", "median", "mode", "min", "max"
        name: str = "llm_reward",
        weight: float = 1.0,
        system_prompt: str = "You are an AI assistant that evaluates text quality.",
        timeout: float = 30.0,
        default_score: float = 0.5,
    ):
        """
        Initialize the LLM reward function.

        Args:
            :param api_url: URL of the OpenAI-compatible API endpoint
            :param api_key: API key for authentication
            :param prompt_template: Jinja2 template for the prompt (has access to 'completion' and 'prompt' variables)
            :param model: Model name to use
            :param temperature: Temperature for sampling (lower = more deterministic)
            :param max_tokens: Maximum number of tokens in the response
            :param output_parser: Function to parse the LLM output into a float score
            :param max_concurrent_requests: Maximum number of concurrent API requests
            :param max_retries: Maximum number of retries for failed requests
            :param retry_base_delay: Base delay for exponential backoff (seconds)
            :param retry_max_delay: Maximum delay for exponential backoff (seconds)
            :param ensemble_size: Number of times to grade each completion
            :param ensemble_aggregation: Type of aggregation function to use to produce the final score for ensemble
            :param name: Name of this reward function
            :param weight: Weight of this reward function
            :param system_prompt: System prompt to set the context for the LLM
            :param timeout: Timeout for API requests (seconds)
            :param default_score: Default score to use if parsing fails
        """
        super().__init__(name=name, weight=weight)
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.default_score = default_score

        if isinstance(api_key, (list, tuple)):
            self.key_manager = APIKeyManager(api_key)
        else:
            self.key_manager = APIKeyManager([api_key])

        # score each completion this many times
        self.ensemble_size = max(1, ensemble_size)
        # what do with all those scores
        self.ensemble_aggregation = ensemble_aggregation

        self.template_env = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.prompt_template = self.template_env.from_string(prompt_template)
        self.output_parser = output_parser or LLMReward._default_output_parser
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore: Optional[Semaphore] = None

        # retry stuff
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @staticmethod
    def _default_output_parser(text: str) -> float:
        """
        Default parser that looks for a score in the format "Score: X.X" or "Rating: X.X/10".

        Args:
            :param text: The LLM response text

        Returns:
            :return A float score
        """
        try: # first, we look for json with `score` field
            data = json.loads(text)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                # Normalize to 0-1 if needed
                if score > 1.0:
                    score = score / 100.0 # arbitrary
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # look for "Score: X.X/Y.Y" pattern
        match = score_pattern.search(text)

        if match:
            score = float(match.group(1))
            max_score = float(match.group(2)) if match.group(2) else 100.0
            normalized_score = score / max_score
            return max(0.0, min(1.0, normalized_score))

        # give up
        logger.warning(f"Could not parse score from LLM response: {text}")
        raise ValueError("Could not parse score from LLM response") # retry

    async def _make_api_request(self, prompt: str, completion: str, session: aiohttp.ClientSession) -> float:
        """
        Make an API request to the LLM and parse the response.

        Args:
            :param prompt: The user prompt
            :param completion: The model completion to evaluate
            :param session: The aiohttp session to use

        Returns:
            :return A float score between 0.0 and 1.0
        """

        formatted_prompt = self.prompt_template.render(
            prompt=prompt,
            completion=completion,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False,
        }

        @retry(
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)),
            wait=wait_exponential(multiplier=self.retry_base_delay, max=self.retry_max_delay),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying API request (attempt {retry_state.attempt_number}/{self.max_retries})"
            ),
            stop=stop_after_attempt(self.max_retries) if self.max_retries > 0 else None
        )
        async def _make_request():
            async with self._semaphore:
                api_key = self.key_manager.get_next_key()

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                try:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 429:
                            # oops we rate limited! Mark key as limited and retry
                            retry_after = float(response.headers.get("Retry-After", 60))
                            self.key_manager.mark_rate_limited(api_key, retry_after)
                            response.raise_for_status() # retry

                        response.raise_for_status()

                        result = await response.json()

                        if "choices" in result and len(result["choices"]) > 0:
                            if "message" in result["choices"][0]:
                                content = result["choices"][0]["message"]["content"]
                            else:
                                content = result["choices"][0]["text"]
                        else:
                            logger.error(f"Unexpected API response format: {result}")
                            raise ValueError("Unexpected API response format")

                        score = self.output_parser(content)
                        return score

                except aiohttp.ClientResponseError as e:
                    logger.error(f"API request failed with status {e.status}: {e.message}")
                    if e.status == 429:
                        retry_after = float(e.headers.get("Retry-After", self.retry_max_delay))
                        self.key_manager.mark_rate_limited(api_key, retry_after)
                        logger.warning(f"Rate limited. Waiting for {retry_after} seconds.")
                        await asyncio.sleep(1)  # Short sleep before trying the next key
                    raise ValueError("Rate limited, trying next key", e)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"API request failed: {str(e)}")
                    await asyncio.sleep(random.uniform(0, 2)) # jitter
                    raise e
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse API response: {str(e)}")
                    raise ValueError("Failed to parse API response", e)
                except Exception as e:
                    logger.error(f"Unexpected error during API request: {str(e)}")
                    raise ValueError("Unexpected error during API request", e)

        try:
            return await _make_request()
        except Exception as e:
            logger.error(f"All retries failed for API request: {str(e)}", e)
            return self.default_score

    def _aggregate_scores(self, scores: List[float]) -> float:
        """
        Aggregate multiple scores from ensemble members.

        Args:
            :param scores: List of scores from ensemble members

        Returns:
            :return Aggregated score
        """
        if not scores:
            return self.default_score

        if len(scores) == 1:
            return scores[0]

        if self.ensemble_aggregation == "mean":
            return float(np.mean(scores))
        elif self.ensemble_aggregation == "median":
            return float(np.median(scores))
        elif self.ensemble_aggregation == "mode":
            rounded_scores = [round(s * 10) / 10 for s in scores]
            return float(statistics.mode(rounded_scores))
        elif self.ensemble_aggregation == "min":
            return float(min(scores))
        elif self.ensemble_aggregation == "max":
            return float(max(scores))
        else:
            logger.warning(f"Unknown aggregation method: {self.ensemble_aggregation}. Using median.")
            return float(np.median(scores))

    async def _evaluate_completion_ensemble(
        self,
        prompt: str,
        completion: str,
        session: aiohttp.ClientSession,
    ) -> float:
        """
        Evaluate a single completion using an ensemble of LLM calls.

        Args:
            :param prompt: The prompt that generated the completion
            :param completion: Completion to evaluate
            :param session: The aiohttp session to use

        Returns:
            :return Aggregated score from the ensemble
        """
        tasks = []
        for i in range(self.ensemble_size):
            task = asyncio.create_task(self._make_api_request(prompt, completion, session))
            tasks.append(task)

        ensemble_scores = []
        for task in asyncio.as_completed(tasks):
            try:
                score = await task
                ensemble_scores.append(score)
            except Exception as e:
                logger.error(f"Ensemble member failed: {str(e)}")
                ensemble_scores.append(self.default_score)

        return self._aggregate_scores(ensemble_scores)

    async def _evaluate_batch(self, completions: List[str], prompts: List[str]) -> List[float]:
        """
        Evaluate a batch of completions using Openai-compat API.

        Args:
            :param completions: List of completions to evaluate
            :param prompts: List of prompts that generated the completions

        Returns:
            :return List of scores in the same order as completions
        """
        self.raw_scores = {} # clear old shit
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async with aiohttp.ClientSession() as session:
            coroutines = [
                self._evaluate_completion_ensemble(prompt, completion, session)
                for completion, prompt in zip(completions, prompts)
            ]

            try:
                results = await asyncio.gather(*coroutines, return_exceptions=True)

                # exceptions are replaced with default score
                scores = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task failed for completion {i}: {str(result)}")
                        scores.append(self.default_score)
                    else:
                        scores.append(result)

                return scores

            except Exception as e:
                logger.error(f"Failed to evaluate batch: {str(e)}")
                return [self.default_score] * len(completions)

    @timed_execution
    def __call__(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Calculate reward scores for completions using the LLM API.

        Args:
            :param completions: List of model outputs to evaluate
            :param prompts: List of prompts used to generate completions
            :param **kwargs: Additional arguments

        Returns:
            :return List of reward scores
        """

        try:
            return asyncio.run(self._evaluate_batch(completions, prompts))

        except Exception as e:
            logger.error(f"Failed to evaluate batch: {str(e)}")
            scores = [self.default_score] * len(completions)

        return scores
