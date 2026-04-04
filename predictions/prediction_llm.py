from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model
from config.global_config import ModelType, SentimentLabel
from predictions.prediction_model_base import PredictionModel


def _aspect_sentiment_model(aspects: tuple[str, ...]) -> type[BaseModel]:
    """JSON schema with one field per aspect; each value is a SentimentLabel enum."""
    fields = {
        name: (
            SentimentLabel,
            Field(description=f"Sentiment for aspect {name!r}."),
        )
        for name in aspects
    }
    return create_model(
        "AspectSentimentOutput",
        __config__=ConfigDict(use_enum_values=True, extra="forbid"),
        **fields,
    )


class LLMPredictionModel(PredictionModel):
    _DEFAULT_SYSTEM = (
        "For each aspect field in the response, output exactly one sentiment label "
        "for the user review: positive, neutral, negative, or notmentioned "
        "(lowercase, as in the schema)."
    )

    def __init__(
        self,
        aspects: list[str] | None = None,
        openai_model: str = "gpt-4o-mini",
        system_prompt: str = "",
        api_key: str | None = None,
    ):
        if api_key is None:
            raise ValueError("API key is required for LLM model")

        aspects = aspects if aspects is not None else []

        super().__init__(ModelType.LLM, aspects)
        self.client = OpenAI(api_key=api_key)
        self.model = openai_model
        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM
        self._response_model = _aspect_sentiment_model(tuple(aspects))

    def predict(self, text: str) -> dict[str, str]:
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=self._response_model,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            raise RuntimeError("LLM returned no structured output")

        out = parsed.model_dump(mode="python")

        return {k: str(v) for k, v in out.items()}
