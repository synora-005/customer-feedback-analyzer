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
MODEL_DIR = "data/model"          # saved model; data/ is git-ignored because the model is about 500 MB


def make_dataset(examples, tokenizer, max_comment_tokens=60):
    """Encode every example. A plain list works as a dataset: it has a length and can be indexed."""
    return [encode_example(e["comment"], e["label"], tokenizer, max_comment_tokens) for e in examples]


def pad_batch(features, pad_id):
    """Make one batch: shorter examples are padded on the right. Padding is ignored by attention and by the loss."""
    import torch
    width = max(len(f["input_ids"]) for f in features)

    def pad(key, value):
        return torch.tensor([f[key] + [value] * (width - len(f[key])) for f in features])

    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)}


def label_accuracy(predictions, labels):
    """Share of label tokens predicted correctly.

    The model's prediction at position i is for position i + 1, so the predictions are shifted
    by one before comparing. Positions marked -100 are ignored.
    """
    import numpy as np
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    guessed, wanted = predictions[:, :-1], labels[:, 1:]
    counted = wanted != -100
    if not counted.any():
        return 0.0
    return float((guessed[counted] == wanted[counted]).mean())


def _most_likely_token(logits, labels):
    """Keep only the best token at each position, so evaluation does not store 50,000 scores per position."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _compute_metrics(eval_prediction):
    return {"accuracy": label_accuracy(eval_prediction.predictions, eval_prediction.label_ids)}


def train_classifier(model, tokenizer, train, val=None, output_dir="data/trainer_tmp",
                     epochs=5, lr=1e-4, batch_size=8, seed=0, logging_steps=5):
    """Fine-tune with the Hugging Face Trainer. Returns the loss and accuracy after each log step or epoch."""
    from functools import partial
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    train_data = make_dataset(train, tokenizer)
    val_data = make_dataset(val, tokenizer) if val else None
    arguments = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        eval_strategy="epoch" if val_data else "no", save_strategy="no",
        logging_steps=logging_steps, report_to="none", seed=seed,
        remove_unused_columns=False,                      # our examples are plain dictionaries
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_data, eval_dataset=val_data,
        data_collator=partial(pad_batch, pad_id=tokenizer.pad_token_id),
        compute_metrics=_compute_metrics if val_data else None,
        preprocess_logits_for_metrics=_most_likely_token if val_data else None,
    )
    trainer.train()
    log = trainer.state.log_history
    return {
        "train_loss": [entry["loss"] for entry in log if "loss" in entry],
        "eval_loss": [entry["eval_loss"] for entry in log if "eval_loss" in entry],
        "eval_accuracy": [entry["eval_accuracy"] for entry in log if "eval_accuracy" in entry],
    }


def save_classifier(model, tokenizer, path=MODEL_DIR):
    """Save the model and the tokenizer (with its new tokens) together, so milestone 4 can load both."""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Develop an inference pipeline to classify new feedback using your fine-tuned model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Build a simple Streamlit UI to interact with your classification model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")