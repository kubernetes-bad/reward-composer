import asyncio
import json
import logging
import random
import re
import statistics
from asyncio import Semaphore
from typing import Union, List, Sequence, Optional, Callable

import aiohttp
import jinja2
import numpy as np
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt, stop_never

from .rewards import Message
from .async_reward import AsyncReward
from .llm_reward import APIKeyManager

logger = logging.getLogger("AsyncLLMReward")

# for default score parser. You're encouraged to write your own parser
score_float_pattern_str = r"(?:score|rating):\s*(?:\[)?(\d+(?:\.\d+)?)(?:\s*\/\s*(\d+(?:\.\d+)?))?(?:\])?"
score_pattern = re.compile(score_float_pattern_str, re.IGNORECASE)

class AsyncLLMReward(AsyncReward):
    def __init__(self,
        api_url: str,
        api_key: Union[str, Sequence[str]],
        prompt_template: str,
        model: str = "Oniichat/Onii-1.3-13b", # come on, set your model
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        output_parser: Optional[Callable[[str], float]] = None, # please use one!
        prompt_preprocessor: Optional[Callable[[Union[List[Message], str], ...], str]] = None,
        completion_preprocessor: Optional[Callable[[Union[List[Message], str], ...], str]] = None,
        max_concurrent_requests: int = 10,
        max_retries: int = 10, # set to 0 for infinite retries
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
        super().__init__(name=name, weight=weight, default_score=default_score)
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.default_score = default_score
        self.max_concurrent_requests = max_concurrent_requests

        keys = api_key if isinstance(api_key, list) else [api_key]
        self.key_manager = APIKeyManager(keys)

        # score each completion this many times
        self.ensemble_size = max(1, ensemble_size)
        # what to do with all those scores
        self.ensemble_aggregation = ensemble_aggregation

        template_env = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.prompt_template = template_env.from_string(prompt_template)
        self.output_parser = output_parser or self._default_output_parser

        self.prompt_preprocessor = prompt_preprocessor or self._default_preprocessor
        self.completion_preprocessor = completion_preprocessor or self._default_preprocessor

        # retry stuff
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def __call__(self, completions: List[Union[List[Message], str]], prompts: List[Union[List[Message], str]], **kwargs):
        self._semaphore = Semaphore(self.max_concurrent_requests)
        return super().__call__(completions=completions, prompts=prompts, **kwargs)

    async def _evaluate_batch(self,
        completions: List[List[Union[Message, str]]],
        prompts: List[List[Union[Message, str]]],
        **kwargs,
    ):
        async with aiohttp.ClientSession() as session:
            return await super()._evaluate_batch(completions=completions, prompts=prompts, session=session, **kwargs)


    @staticmethod
    def _default_output_parser(text: str) -> float:
        """
        Default parser that looks for a score in the format "Score: X.X" or "Rating: X.X/10".

        Args:
            :param text: LLM response text

        Returns:
            :return score that was found in LLM response
        """
        try:  # first, we look for json with `score` field
            data = json.loads(text)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                max_score = float(data.get("max_score", 100))  # arbitrary
                # Normalize to 0-1 if needed
                if score > 1.0:
                    score = score / max_score
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # that's ok - try other things below

        # next, look for "Score: X.X/Y.Y" pattern
        match = score_pattern.search(text)

        if match:
            score = float(match.group(1))
            max_score = float(match.group(2)) if match.group(2) else 100.0
            normalized_score = score / max_score
            return max(0.0, min(1.0, normalized_score))

        # give up
        raise ValueError(f"Could not parse score from LLM response {text}") # retry or default score

    @staticmethod
    def _default_preprocessor(turn: Union[List[Message], str], **kwargs) -> str:
        """just extracts content from last message"""
        if isinstance(turn, str):
            return turn
        # get the last message content
        if len(turn) == 0:
            return "" # it's an empty convo bro
        return turn[-1].content

    async def _make_api_request(self, user_text: str, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Make an API request to the LLM and parse the response.

        Args:
            :param user_text: The contents of user turn sent to the model (same thing as Task message).
            :param session: The aiohttp session to use

        Returns:
            :return Response from the LLM
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": max(0.0, min(self.temperature + random.uniform(-0.05, 0.05), 1.0)), # jitter to overcome prompt caching
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
            stop=stop_after_attempt(self.max_retries) if self.max_retries > 0 else stop_never
        )
        async def _make_request() -> Optional[str]:
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
                                content: str = result["choices"][0]["message"]["content"]
                            else:
                                content: str = result["choices"][0]["text"]
                        else:
                            logger.error(f"Unexpected API response format: {result}")
                            raise ValueError("Unexpected API response format")

                        return content

                except aiohttp.ClientResponseError as e:
                    logger.error(f"API request failed with status {e.status}: {e.message}")
                    if e.status == 429:
                        retry_after = float(e.headers.get("Retry-After", self.retry_max_delay))
                        self.key_manager.mark_rate_limited(api_key, retry_after)
                        logger.warning(f"Rate limited. Waiting for {retry_after} seconds.")
                        await asyncio.sleep(1)  # Short sleep before trying the next key
                        raise ValueError("Rate limited, trying next key")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"API request failed: {str(e)}")
                    await asyncio.sleep(random.uniform(0, 2)) # jitter
                    raise e
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse API response: {str(e)}")
                    raise ValueError("Failed to parse API response")
                except Exception as e:
                    logger.error(f"Unexpected error during API request: {str(e)}")
                    raise ValueError("Unexpected error during API request")

        try:
            return await _make_request()
        except Exception as e:
            logger.error(f"All retries failed for API request: {str(e)}", e)
            return None

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

    async def evaluate_one(self, completion: Union[List[Message], str], prompt: Union[List[Message], str], **kwargs) -> float:
        session = kwargs.get("session")
        if session is None:
            session = aiohttp.ClientSession()

        preprocessed_prompt = self.prompt_preprocessor(prompt, **kwargs)
        preprocessed_completion = self.completion_preprocessor(completion, **kwargs)

        # prepare llm prompt
        llm_prompt_text = self.prompt_template.render({
            "prompt": preprocessed_prompt,
            "completion": preprocessed_completion,
            **kwargs,
        })

        # make llm call (multiple in parallel if ensemble scoring
        tasks = []
        for i in range(self.ensemble_size):
            task = asyncio.create_task(self._make_api_request(llm_prompt_text, session))
            tasks.append(task)

        llm_results: List[Optional[Union[str, BaseException]]] = await asyncio.gather(*tasks, return_exceptions=True)
        # replace exceptions with Nones
        for i, llm_result_or_exception in enumerate(llm_results):
            if isinstance(llm_result_or_exception, Exception):
                logger.error(f"Task failed for completion {i}: {str(llm_result_or_exception)}")
                llm_results[i] = None

        ensemble_scores = [  # parse results
            self.output_parser(llm_result) if llm_results is not None
            else self.default_score
            for llm_result in llm_results
        ]

        # ensemble: do the math for final result
        return self._aggregate_scores(ensemble_scores)
