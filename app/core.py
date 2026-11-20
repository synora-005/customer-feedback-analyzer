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
def label_token(label):
    """The special token for a category, for example <pricing>."""
    if label not in CATEGORIES:
        raise ValueError(f"unknown label {label!r}, expected one of {CATEGORIES}")
    return f"<{label}>"


def load_gpt2(name="gpt2"):
    """Download (the first time) and load GPT-2. Returns the tokenizer and the model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer   # imported here because it is slow
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return tokenizer, model


def add_category_tokens(tokenizer, model):
    """Teach the tokenizer and the model the category tokens, plus a padding token.

    Returns how many tokens were new, so doing it twice adds nothing the second time.
    """
    added = tokenizer.add_tokens([label_token(label) for label in CATEGORIES], special_tokens=True)
    if tokenizer.pad_token is None:                # GPT-2 has none, and batches need one
        added += tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))   # one new row of numbers per new token
    model.config.pad_token_id = tokenizer.pad_token_id
    return added


def encode_prompt(comment, tokenizer, max_comment_tokens=60):
    """The text the model reads before it answers: 'Comment: ...' then a new line and 'Category:'.

    Long comments are cut to max_comment_tokens. Training and predicting both use this function.
    """
    head = tokenizer("Comment:")["input_ids"]
    body = tokenizer(" " + comment)["input_ids"][:max_comment_tokens]
    tail = tokenizer("\nCategory:")["input_ids"]
    return head + body + tail


def encode_example(comment, label, tokenizer, max_comment_tokens=60):
    """One training example: the prompt, then the label token the model should answer with.

    In `labels`, -100 means "ignore this position", so only the label token is learned from.
    """
    prompt = encode_prompt(comment, tokenizer, max_comment_tokens)
    label_id = tokenizer.convert_tokens_to_ids(label_token(label))
    return {
        "input_ids": prompt + [label_id],
        "attention_mask": [1] * (len(prompt) + 1),
        "labels": [-100] * len(prompt) + [label_id],
    }


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