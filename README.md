# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/37eg33wsjj/badge.svg)](https://ask.thili.ai/b/37eg33wsjj)

This Customer Feedback Analyzer is a tool for retail managers. It uses a fine-tuned GPT-2 model to automatically categorize customer feedback comments, such as 'product issue' or 'staff praise'. This helps managers quickly identify trends and prioritize improvements based on customer input.

## What you are building

**Goal.** You will build a Customer Feedback Analyzer that automatically categorizes customer comments to help retail managers quickly identify trends and prioritize improvements.

**Done when.** You will have a fine-tuned GPT-2 model that can classify new customer comments by generating special label tokens, and an inference pipeline to make predictions. The project is complete when a Streamlit app allows users to input comments and view their classifications.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Use data preparation techniques to create a small dataset of labeled examples.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Apply tokenizer and model extension methods to incorporate new category tokens.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Utilize the Hugging Face Trainer for fine-tuning your model.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Develop an inference pipeline to classify new feedback using your fine-tuned model.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Build a simple Streamlit UI to interact with your classification model.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more diverse comments and additional categories to improve model robustness.
- Implement batch processing functionality to classify multiple feedback comments simultaneously.
- Display the model's confidence score alongside each classification to provide more insight.
