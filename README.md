# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/37eg33wsjj/badge.svg)](https://ask.thili.ai/b/37eg33wsjj)

This Customer Feedback Analyzer is a tool for retail managers. It uses a fine-tuned GPT-2 model to automatically categorize customer feedback comments, such as 'product issue' or 'staff praise'. This helps managers quickly identify trends and prioritize improvements based on customer input.

## What you are building

**Goal.** You will build a Customer Feedback Analyzer that automatically categorizes customer comments to help retail managers quickly identify trends and prioritize improvements.

**Done when.** You will have a fine-tuned GPT-2 model that can classify new customer comments by generating special label tokens, and an inference pipeline to make predictions. The project is complete when a Streamlit app allows users to input comments and view their classifications.

## Milestones

- [x] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**
- [x] **2. Extend GPT-2 tokenizer and model with new category tokens.**
- [x] **3. Fine-tune the model using the Hugging Face Trainer.**
- [x] **4. Implement an inference function to classify new feedback.**
- [x] **5. Build a simple Streamlit app to input comments and show classifications.**

## How it works

The five categories are `product_issue`, `staff_praise`, `staff_complaint`, `pricing` and `store_experience`.

1. **New tokens.** Five new tokens, `<product_issue>`, `<staff_praise>` and so on, are added to the vocabulary of GPT-2 (the 124M-parameter `gpt2` model). Each starts from the average embedding of the words in its name, not from random numbers.
2. **Prompt.** Each comment becomes `Comment: <text>` followed by a new line and `Category:`. The model is trained to answer with the right label token.
3. **Training.** The Hugging Face Trainer fine-tunes the model, with a loss that compares only the five category scores with each other. The model learns to tell categories apart instead of learning to ignore the other 50,000 words.
4. **Prediction.** For a new comment, the model's scores for the five label tokens become probabilities. The highest is the answer and its probability is the confidence.

## Dataset

- 200 comments, 40 per category, in `dataset/feedback.csv`.
- **The comments were written by an AI assistant (Claude) as a starter dataset, not collected from real customers.** They are cleaner and more obvious than real feedback, so the accuracy below is probably higher than it would be on real comments.
- Duplicates are removed on loading. The data is split with a fixed seed into 140 training, 30 validation and 30 test comments, with each category in the same proportion in every split.
- The test set is used only for the final score. Nothing was tuned on it.

## Results

| | Accuracy |
|---|---|
| Guessing (5 categories) | 0.20 |
| Validation (30 comments), final epoch | 0.867 (26/30) |
| **Test (30 comments)** | **0.900 (27/30)** |

Validation accuracy per epoch: 0.767, 0.833, 0.833, 0.800, 0.867, 0.867, 0.867, 0.867, 0.867, 0.867.

Test accuracy per category (6 comments each): product_issue 1.00, staff_praise 1.00, pricing 1.00, staff_complaint 0.83, store_experience 0.67.

The three test mistakes:

| Real | Model said | Confidence | Comment |
|---|---|---|---|
| store_experience | pricing | 0.40 | "Everything I wanted was out of stock again." |
| store_experience | product_issue | 0.95 | "Trolleys are broken and the wheels are stuck." |
| staff_complaint | staff_praise | 0.99 | "Security followed me around the shop like I was a thief." |

The first two are borderline: stock levels and broken trolleys could reasonably go in either category. The third is a real failure, because the model was almost certain and wrong.

Training settings: 10 epochs, learning rate 5e-5, batch size 8, seed 0, about 6.6 minutes on a laptop CPU. The numbers are saved in `results/training.json` and `results/evaluation.json`.

### What went wrong first

The first two training runs did not work. With a plain loss over the whole vocabulary and randomly started tokens, validation accuracy stayed at exactly 0.2 (chance). The 15-epoch run at learning rate 2e-4 ended with a loss of 1.609, which is ln(5), the loss of a model that treats all five categories as equally likely. Two changes fixed it: starting each new token from the words in its name, and computing the loss over the five category tokens only. Validation accuracy was 0.767 after the first epoch of the fixed run.

## Limits

- **Small test set.** 30 comments means each one is worth about 3 points. A score of 27/30 is consistent with a true accuracy anywhere from roughly 74% to 97%.
- **Synthetic data.** See above. I have not tested it on real customer comments.
- **Overconfident mistakes.** The model can be wrong with a confidence of 0.99, so confidence is not a reliable warning.
- **One label per comment.** Comments with mixed feelings ("lovely staff but far too expensive") get a single category.
- **Only five categories, English only.** Comments are cut to 60 tokens.
- With 6 test comments per category, the differences between categories (such as `store_experience` scoring lowest) are not reliable.

## Run it

Developed on Python 3.14.3 on Windows (PowerShell). The course template targets Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train_it.py --epochs 10 --lr 5e-5   # trains and saves the model to data/model
python evaluate_it.py                       # scores the test set once
python -m streamlit run streamlit_app.py    # the app
pytest                                      # the tests (they use a tiny untrained model)
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The trained model is about 500 MB and is not in the repository (`data/` is git-ignored), so run `train_it.py` first. The app shows a message instead of crashing if the model is missing. The app does not use `.env` or any API key, because the model runs on your own computer.

## Project layout

- `app/core.py`: dataset loading and splitting, new tokens, training, prediction and evaluation
- `train_it.py`, `evaluate_it.py`, `streamlit_app.py`: the three scripts
- `dataset/feedback.csv`: the labelled comments
- `results/`: training and evaluation numbers
- `tests/`: tests for every milestone, run on every push by GitHub Actions

## Stretch ideas

- [ ] Expand the dataset with more diverse comments and additional categories to improve model robustness. Next step: 10 to 20 messy comments per category in my own words, scored as a separate second evaluation.
- [x] Classify multiple feedback comments simultaneously (one comment per line in the app).
- [x] Display the model's confidence score alongside each classification (the app also marks answers under 60% as "unsure").

## How this was made

Built as an askThili course project, with an AI assistant (Claude) helping with the code and writing the starter dataset.