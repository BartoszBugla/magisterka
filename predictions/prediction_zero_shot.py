from transformers import pipeline

from predictions.prediction_model_base import PredictionModel


class ZeroShotPredictionModel(PredictionModel):
    def __init__(
        self,
        aspects: list[str] | None = None,
        hf_model_name: str = "facebook/bart-large-mnli",
    ):
        aspects = aspects if aspects is not None else []
        super().__init__(aspects)

        self.classifier = pipeline(
            "zero-shot-classification", model=hf_model_name, device="mps"
        )
        self.candidate_labels = ["positive", "neutral", "negative", "notmentioned"]

    def predict(self, text: str) -> dict[str, str]:
        results = {}
        for aspect in self.aspects:
            hypothesis_template = f"The sentiment for the {aspect} is {{}}."

            output = self.classifier(
                text,
                self.candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False,  # We only want one dominant sentiment per aspect
            )
            results[aspect] = output["labels"][0]

        return results
