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


def _start_from_words(tokenizer, model):
    """Start each category token at the average embedding of the words in its name, e.g. 'staff praise'."""
    import torch
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for category in CATEGORIES:
            word_ids = tokenizer(" " + category.replace("_", " "))["input_ids"]
            weight[tokenizer.convert_tokens_to_ids(label_token(category))] = weight[word_ids].mean(dim=0)


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
    if added:
        _start_from_words(tokenizer, model)        # a meaningful starting point instead of random numbers
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



def _compute_metrics(eval_prediction):
    return {"accuracy": label_accuracy(eval_prediction.predictions, eval_prediction.label_ids)}


def train_classifier(model, tokenizer, train, val=None, output_dir="data/trainer_tmp",
                     epochs=10, lr=5e-5, batch_size=8, seed=0, logging_steps=5):
    """Fine-tune with the Hugging Face Trainer. Returns the loss and accuracy after each log step or epoch.

    The loss compares only the five category tokens with each other, so the model spends its effort
    on telling the categories apart instead of on ignoring the other 50,000 words.
    """
    import torch
    import torch.nn.functional as F
    from functools import partial
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    category_ids = torch.tensor([tokenizer.convert_tokens_to_ids(label_token(c)) for c in CATEGORIES])

    class ClassifierTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            ids = category_ids.to(outputs.logits.device)
            targets = labels[:, 1:]                                  # the prediction at i is for i + 1
            wanted = targets != -100
            scores = outputs.logits[:, :-1][wanted][:, ids]          # one score per category
            answer = (targets[wanted].unsqueeze(1) == ids.unsqueeze(0)).long().argmax(dim=1)
            loss = F.cross_entropy(scores, answer)
            return (loss, outputs) if return_outputs else loss

    def best_category(logits, labels):
        """At every position, the category token with the highest score (so only categories can be 'predicted')."""
        if isinstance(logits, tuple):
            logits = logits[0]
        ids = category_ids.to(logits.device)
        return ids[logits[..., ids].argmax(dim=-1)]

    train_data = make_dataset(train, tokenizer)
    val_data = make_dataset(val, tokenizer) if val else None
    arguments = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        eval_strategy="epoch" if val_data else "no", save_strategy="no",
        logging_steps=logging_steps, report_to="none", seed=seed,
        remove_unused_columns=False,                      # our examples are plain dictionaries
    )
    trainer = ClassifierTrainer(
        model=model, args=arguments, train_dataset=train_data, eval_dataset=val_data,
        data_collator=partial(pad_batch, pad_id=tokenizer.pad_token_id),
        compute_metrics=_compute_metrics if val_data else None,
        preprocess_logits_for_metrics=best_category if val_data else None,
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
def load_classifier(path=MODEL_DIR):
    """Load the model and tokenizer saved by save_classifier. Returns the tokenizer and the model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict_label(comment, model, tokenizer, max_comment_tokens=60):
    """Classify one comment.

    Reads the model's scores for the five category tokens only (the model could in theory answer
    with any word, but a label must be one of the categories) and turns them into probabilities.
    Returns {"label": best category, "confidence": its probability, "scores": probability of each}.
    """
    import torch
    if not comment or not comment.strip():
        raise ValueError("the comment is empty")
    prompt = encode_prompt(" ".join(comment.split()), tokenizer, max_comment_tokens)
    category_ids = [tokenizer.convert_tokens_to_ids(label_token(c)) for c in CATEGORIES]

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt], device=device)).logits[0, -1]
    if was_training:
        model.train()

    probabilities = torch.softmax(logits[category_ids], dim=-1).tolist()
    scores = dict(zip(CATEGORIES, probabilities))
    best = max(scores, key=scores.get)
    return {"label": best, "confidence": scores[best], "scores": scores}


def evaluate_classifier(examples, model, tokenizer):
    """Score the model on labelled examples (use the test set, once, at the end).

    Returns the accuracy, the accuracy per category, a confusion table (real label -> what the
    model said) and the list of mistakes.
    """
    confusion = {real: {said: 0 for said in CATEGORIES} for real in CATEGORIES}
    mistakes = []
    for example in examples:
        result = predict_label(example["comment"], model, tokenizer)
        confusion[example["label"]][result["label"]] += 1
        if result["label"] != example["label"]:
            mistakes.append({"comment": example["comment"], "real": example["label"],
                             "said": result["label"], "confidence": round(result["confidence"], 3)})
    total = len(examples)
    correct = total - len(mistakes)
    per_label = {}
    for label in CATEGORIES:
        count = sum(confusion[label].values())
        per_label[label] = round(confusion[label][label] / count, 3) if count else None
    return {
        "examples": total, "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "chance": round(1 / len(CATEGORIES), 3),
        "per_label_accuracy": per_label, "confusion": confusion, "mistakes": mistakes,
    }


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Build a simple Streamlit UI to interact with your classification model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")