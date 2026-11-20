"""Score the saved model once on the untouched test set: `python evaluate_it.py`."""
import json
from pathlib import Path

from app.core import MODEL_DIR, evaluate_classifier, load_classifier, load_feedback, split_dataset

if __name__ == "__main__":
    train, val, test = split_dataset(load_feedback("dataset/feedback.csv"))   # same seed as training
    tokenizer, model = load_classifier(MODEL_DIR)
    report = evaluate_classifier(test, model, tokenizer)

    print(f"test accuracy: {report['accuracy']} ({report['correct']}/{report['examples']}), "
          f"guessing would give about {report['chance']}")
    print("accuracy per category:", report["per_label_accuracy"])
    print("mistakes:")
    for m in report["mistakes"]:
        print(f"  real={m['real']}  said={m['said']} ({m['confidence']})  {m['comment']}")

    Path("results").mkdir(exist_ok=True)
    Path("results/evaluation.json").write_text(json.dumps(report, indent=2))
    print("saved results/evaluation.json")