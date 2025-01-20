from slalom_explanations import SLALOMLocalExplanantions
from experiment_utils.reference_models import TFIDFModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_dataset, Dataset

def get_small_imdb():
    """ Helper function to get IMDB dataset. """
    imdb = load_dataset('imdb', verification_mode='no_checks').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
    imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
    imdb["test"] = imdb["test"].shuffle(seed=42).select(range(50))
    return imdb

def test_initialization_w_tfidf():
    """ Compute a dummy explanation for a tfidf model. """
    imdbdata = get_small_imdb()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    mymodel = TFIDFModel(imdbdata, tokenizer)
    slalom_explainer = SLALOMLocalExplanantions(mymodel, tokenizer, modes=["value", "imp"], device="cpu")
    input_example = "This is just an example."
    res = slalom_explainer.tokenize_and_explain(input_example)
    for token, expls in res:
        assert token in input_example.lower()
        assert len(expls) == 2

def test_quickstart_model():
    """ Test the model in the quickstart notebook. """
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    slalom_explainer = SLALOMLocalExplanantions(model, tokenizer, modes=["value", "imp", "lin"], device="cpu")
    example_text = "This movie was so frustrating. everything seemed energetic and i was totally prepared to have a good time. \
i at least thought i\'d be able to stand it. but, i was wrong. first, the weird looping? it was like watching \" america\'s funniest home videos \". \
the damn parents. i hated them so much. the stereo - typical latino family? i need to speak with the person responsible for this. we need to have a talk. \
that little girl who was always hanging on someone? i just hated her and had to mention it. now, the final scene transcends, i must say. \
it\'s so gloriously bad and full of badness that it is a movie of its own. what crappy dancing. horrible and beautiful at once."
    res_explanation = slalom_explainer.tokenize_and_explain(example_text)
    value_fun = None # value score of bad
    value_bad = None # value score of fun
    for token, expls in res_explanation:
        assert len(expls) == 3
        if token == "fun":
            value_fun = expls
        if token == "bad":
            value_bad = expls
    ## Check that "fun" is asserted a higher value than "bad" (same for linear scores)
    assert value_fun[0] > value_bad[0]
    assert value_fun[2] > value_bad[2]

if __name__ == "__main__":
    test_initialization_w_tfidf()
    test_quickstart_model()