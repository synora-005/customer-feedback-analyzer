"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
from pathlib import Path

import pytest
import torch

from app.core import (
    CATEGORIES,
    add_category_tokens,
    dataset_stats,
    encode_example,
    encode_prompt,
    label_token,
    load_feedback,
    split_dataset,
)

DATASET = Path(__file__).parent.parent / "dataset" / "feedback.csv"


def make_examples(per_label=20):
    return [
        {"comment": f"{label} comment {i}", "label": label}
        for label in CATEGORIES
        for i in range(per_label)
    ]


# Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
def test_milestone_1(tmp_path):
    path = tmp_path / "feedback.csv"
    path.write_text(
        "comment,label\n"
        '"The cashier  was lovely, thank you",staff_praise\n'   # extra space, and a comma inside the quotes
        '"the cashier was lovely, thank you",staff_praise\n'    # the same comment once tidied
        ",pricing\n"                                             # empty comment
        '"Prices are too high",pricing\n',
        encoding="utf-8",
    )
    assert load_feedback(str(path)) == [
        {"comment": "The cashier was lovely, thank you", "label": "staff_praise"},
        {"comment": "Prices are too high", "label": "pricing"},
    ]

    # a label we do not know about is an error, not something to quietly accept
    bad = tmp_path / "bad.csv"
    bad.write_text('comment,label\n"Great shop",refunds\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_feedback(str(bad))


def test_milestone_1_split():
    examples = make_examples()
    train, val, test = split_dataset(examples, seed=0)
    assert (len(train), len(val), len(test)) == (70, 15, 15)

    # every category appears in every split
    for split in (train, val, test):
        assert {example["label"] for example in split} == set(CATEGORIES)

    # nothing is in two splits
    comments = [example["comment"] for example in train + val + test]
    assert len(set(comments)) == len(comments) == len(examples)

    # the same seed gives the same split, a different seed a different one
    assert split_dataset(examples, seed=0) == (train, val, test)
    assert split_dataset(examples, seed=1)[0] != train

    # the fractions have to leave something for the test set
    with pytest.raises(ValueError):
        split_dataset(examples, train_fraction=0.9, val_fraction=0.2)


def test_milestone_1_stats():
    stats = dataset_stats([
        {"comment": "a b c", "label": "pricing"},
        {"comment": "d e", "label": "staff_praise"},
    ])
    expected = {label: 0 for label in CATEGORIES}
    expected.update(pricing=1, staff_praise=1)
    assert stats == {"count": 2, "per_label": expected, "avg_words": 2.5}


def test_milestone_1_real_dataset():
    stats = dataset_stats(load_feedback(str(DATASET)))
    assert stats["count"] >= 150
    assert all(count >= 30 for count in stats["per_label"].values())


# Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
# These use GPT-2's real tokenizer (a small download) but a tiny untrained model, so no big download.
@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")     # a fresh copy for every test


def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=2, n_head=2))


def test_milestone_2(tokenizer):
    model = tiny_gpt2()
    original_size = len(tokenizer)

    added = add_category_tokens(tokenizer, model)
    assert added == len(CATEGORIES) + 1                       # the categories and the padding token
    assert len(tokenizer) == original_size + added
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
    assert model.get_output_embeddings().weight.shape[0] == len(tokenizer)

    # each category token is exactly one new token, and it is never split into pieces
    ids = []
    for category in CATEGORIES:
        token_ids = tokenizer(label_token(category))["input_ids"]
        assert len(token_ids) == 1
        assert tokenizer.decode(token_ids) == label_token(category)
        ids.append(token_ids[0])
    assert len(set(ids)) == len(CATEGORIES)
    assert min(ids) >= original_size

    # doing it again changes nothing
    assert add_category_tokens(tokenizer, model) == 0

    # the model accepts the new tokens
    logits = model(torch.tensor([ids])).logits
    assert logits.shape[-1] == len(tokenizer)

    # an unknown category is refused
    with pytest.raises(ValueError):
        label_token("refunds")


def test_milestone_2_encoding(tokenizer):
    add_category_tokens(tokenizer, tiny_gpt2())
    example = encode_example("Great service!", "staff_praise", tokenizer)
    label_id = tokenizer.convert_tokens_to_ids("<staff_praise>")
    ids, labels = example["input_ids"], example["labels"]

    # the label token comes last, and it is the only position the model learns from
    assert ids[-1] == label_id
    assert len(ids) == len(labels) == len(example["attention_mask"])
    assert labels[-1] == label_id
    assert all(value == -100 for value in labels[:-1])

    # everything before the label is the same prompt used when predicting
    assert ids[:-1] == encode_prompt("Great service!", tokenizer)
    text = tokenizer.decode(ids[:-1])
    assert text.startswith("Comment: Great service!")
    assert text.endswith("Category:")

    # long comments are cut: prompt without comment + 20 comment tokens + 1 label token
    long_example = encode_example("word " * 500, "pricing", tokenizer, max_comment_tokens=20)
    assert len(long_example["input_ids"]) == len(encode_prompt("", tokenizer, 0)) + 20 + 1

    with pytest.raises(ValueError):
        encode_example("x", "refunds", tokenizer)


# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")