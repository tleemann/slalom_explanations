from slalom_explanations import SLALOMLocalExplanantions
from experiment_utils.reference_models import TFIDFModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_dataset, Dataset


def get_small_imdb():
    """Helper function to get IMDB dataset."""
    imdb = load_dataset("imdb", verification_mode="no_checks").with_format(
        "torch", device="cpu"
    )  # format to pytorch tensors, but leave data on cpu
    imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
    imdb["test"] = imdb["test"].shuffle(seed=42).select(range(50))
    return imdb


def test_initialization_w_tfidf():
    """Compute a dummy explanation for a tfidf model."""
    imdbdata = get_small_imdb()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    mymodel = TFIDFModel(imdbdata, tokenizer)
    slalom_explainer = SLALOMLocalExplanantions(
        mymodel, tokenizer, modes=["value", "imp"], device="cpu"
    )
    input_example = "This is just an example."
    res = slalom_explainer.tokenize_and_explain(input_example)
    for token, expls in res:
        assert token in input_example.lower()
        assert len(expls) == 2


if __name__ == "__main__":
    test_initialization_w_tfidf()
