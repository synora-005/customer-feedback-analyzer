"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
from pathlib import Path

import pytest

from app.core import CATEGORIES, dataset_stats, load_feedback, split_dataset

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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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