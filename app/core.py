"""Your code. One stub per milestone; build them in order and keep each one small."""

import csv
import random
from collections import defaultdict


# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
# Hint: Use data preparation techniques to create a small dataset of labeled examples.
CATEGORIES = ["product_issue", "staff_praise", "staff_complaint", "pricing", "store_experience"]


def load_feedback(path):
    """Read a CSV file with the columns `comment` and `label`.

    Tidies the spaces in each comment, skips empty comments and repeats (ignoring capital
    letters), and refuses labels that are not in CATEGORIES.
    Returns a list of {"comment": ..., "label": ...} dictionaries.
    """
    examples = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as f:
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            comment = " ".join((row.get("comment") or "").split())
            label = (row.get("label") or "").strip()
            if not comment:
                continue
            if label not in CATEGORIES:
                raise ValueError(f"line {line_number}: unknown label {label!r}, expected one of {CATEGORIES}")
            if comment.lower() in seen:
                continue
            seen.add(comment.lower())
            examples.append({"comment": comment, "label": label})
    return examples


def split_dataset(examples, train_fraction=0.7, val_fraction=0.15, seed=0):
    """Split into train, validation and test sets, keeping each label in the same proportion.

    The rest after train and validation becomes the test set. The same seed gives the same split.
    """
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must leave some examples for the test set")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for example in examples:
        by_label[example["label"]].append(example)

    train, val, test = [], [], []
    for label in sorted(by_label):                 # a fixed order, so the seed always gives the same split
        group = by_label[label][:]
        rng.shuffle(group)
        n_train = round(len(group) * train_fraction)
        n_val = round(len(group) * val_fraction)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]

    for split in (train, val, test):               # mix the labels together inside each split
        rng.shuffle(split)
    return train, val, test


def dataset_stats(examples):
    """A few numbers that describe the dataset."""
    per_label = {label: 0 for label in CATEGORIES}
    words = 0
    for example in examples:
        per_label[example["label"]] += 1
        words += len(example["comment"].split())
    return {
        "count": len(examples),
        "per_label": per_label,
        "avg_words": round(words / len(examples), 1) if examples else 0.0,
    }


# --- Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
# Hint: Apply tokenizer and model extension methods to incorporate new category tokens.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Fine-tune the model using the Hugging Face Trainer.
# Hint: Utilize the Hugging Face Trainer for fine-tuning your model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Develop an inference pipeline to classify new feedback using your fine-tuned model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Build a simple Streamlit UI to interact with your classification model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")