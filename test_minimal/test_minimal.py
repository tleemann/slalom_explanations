## Test quickstart notebook minimal requirements.
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification, BertConfig
from slalom_explanations import SLALOMLocalExplanantions, slalom_highlight_plot
from slalom_explanations import linearize_explanation, slalom_scatter_plot
from slalom_explanations.slalom_helpers import SLALOMModelWrapper

import matplotlib.pyplot as plt


def test_quickstart_model():
    """Test the model in the quickstart notebook."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "lvwerra/distilbert-imdb"
    )
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    slalom_explainer = SLALOMLocalExplanantions(
        model, tokenizer, modes=["value", "imp", "lin"], device="cpu"
    )
    example_text = "This movie was so frustrating. everything seemed energetic and i was totally prepared to have a good time. \
i at least thought i'd be able to stand it. but, i was wrong. first, the weird looping? it was like watching \" america's funniest home videos \". \
the damn parents. i hated them so much. the stereo - typical latino family? i need to speak with the person responsible for this. we need to have a talk. \
that little girl who was always hanging on someone? i just hated her and had to mention it. now, the final scene transcends, i must say. \
it's so gloriously bad and full of badness that it is a movie of its own. what crappy dancing. horrible and beautiful at once."
    res_explanation = slalom_explainer.tokenize_and_explain(example_text)
    value_fun = None  # value score of bad
    value_bad = None  # value score of fun
    for token, expls in res_explanation:
        assert len(expls) == 3
        if token == "fun":
            value_fun = expls
        if token == "bad":
            value_bad = expls
    ## Check that "fun" is asserted a higher value than "bad" (same for linear scores)
    assert value_fun[0] > value_bad[0]
    assert value_fun[2] > value_bad[2]

    highlight_toks = [
        "fun",
        "bad",
        "crap",
        "horrible",
        "someone",
        "the",
        "beautiful",
        "energetic",
        "fun",
        "for",
        "this",
        "person",
        "family",
    ]
    plt.ioff()
    slalom_scatter_plot(
        res_explanation, sizey=4, sizex=4, highlight_toks=highlight_toks, fontsize=8
    )
    linear_scores = linearize_explanation(res_explanation)
    slalom_highlight_plot(linear_scores, vmax=1)


def test_multiclass_model():
    """Test a multiclass untrained BERT model to see if the target_class attribute works."""
    config = BertConfig()
    config.num_labels = 15
    mymodel = BertForSequenceClassification(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Example from https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2024-1-1
    input_sentence = "Human rights and international humanitarian law are fundamental pillars of a secure global system. These principles are under threat. \
Some of the world's most powerful states choose to sell arms to human-rights abusing states. These principles are under threat."
    slalom_explainer = SLALOMLocalExplanantions(
        mymodel, tokenizer, modes=["value", "imp"], target_class=10, device="cpu"
    )
    res_explanation = slalom_explainer.tokenize_and_explain(input_sentence)
    assert isinstance(slalom_explainer.model, SLALOMModelWrapper)
