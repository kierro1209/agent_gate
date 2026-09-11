import json
from collections.abc import Callable
from typing import Protocol

from gate.models import TurnContext


class Judge(Protocol):
    def score(self, turn: TurnContext, memory_block: str) -> float: ...


class FunctionJudge:
    def __init__(self, scorer: Callable[[TurnContext, str], float]) -> None:
        self._scorer = scorer

    def score(self, turn: TurnContext, memory_block: str) -> float:
        value = float(self._scorer(turn, memory_block))
        if not 0 <= value <= 1:
            raise ValueError("judge score must be between 0 and 1")
        return value


class GeminiJudge:
    def __init__(self, api_key: str, model: str, prompt: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model, self._prompt = model, prompt

    def score(self, turn: TurnContext, memory_block: str) -> float:
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._prompt.format(
                user_message=turn.user_message, memory_block=memory_block or "(none)"
            ),
            config={"response_mime_type": "application/json"},
        )
        if not response.text:
            raise ValueError("judge returned no text")
        value = float(json.loads(response.text)["score"])
        if not 0 <= value <= 1:
            raise ValueError("judge score must be between 0 and 1")
        return value
