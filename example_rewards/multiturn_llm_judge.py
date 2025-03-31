from typing import Union, List, Any

import jinja2

from reward_composer import AsyncLLMReward, Message

# THIS IS AN EXAMPLE!
# Use your own system prompt. This is here for demonstration purposes only.
# this will not perform well!
# DO NOT DEPLOY THIS TO PROD AS IS
SYSTEM_PROMPT = """You are an expert dialogue evaluator, skilled at assessing the quality of dialogues.

When asked to evaluate a dialogue sample, output your assessment in this format:
```
Reasoning: [Concise critique of strengths and flaws]
Score: [X/100]
```"""

# please use your own task message! This is given for demonstration purposes only
# this will perform bad! Don't use blindly!
TASK = """Here is a dialogue between User and Assistant. Use the context to evaluate quality of the dialogue.

## Prior dialogue:
{{prompt}}

## Suggested conversation path:
{{completion}}

**Instructions**:  
Score the given suggestion on a scale from 0 to 100. Remember to output your assessment in this format:
```
Reasoning: [your reasoning]
Score: [X/100]
```"""

TASK_PROMPT_TEMPLATE = """<dialogue>
   {{ dialogue }}
   </dialogue>"""

class MultiturnLLMJudge(AsyncLLMReward):
    def __init__(self,
                 api_url: str,
                 api_key: Union[str, List[str]],
                 model: str,
                 max_concurrent_requests: int = 10,
                 weight: float = 1.0,
                 ):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            model=model,
            name="multiturn_llm_reward",
            weight=weight,
            max_concurrent_requests=max_concurrent_requests,
            system_prompt=SYSTEM_PROMPT,
            prompt_template=TASK,
            output_parser=self._output_parser,
            temperature=0.0,
            top_p=0.9,
            ensemble_size=3,
            ensemble_aggregation="mean",
        )
        self.prompt_preprocessor = self._prompt_preprocessor

    def _prompt_preprocessor(self, prompt: Union[List[Message], str], **kwargs: Any) -> str:
        contents = ""

        if isinstance(prompt, List) and len(prompt) > 0:
            for message in prompt:
                contents = f"{contents}\n{"User" if message.role == "user" else "Assistant"}: {message.content}\n".strip()
        else:
            contents = prompt.replace("{{user}}", "User") # example usage
        return jinja2.Template(TASK_PROMPT_TEMPLATE).render(dialogue=contents)

    @staticmethod
    def _output_parser(output: str) -> float:
        return AsyncLLMReward._default_output_parser(output)
