from app.core import dataset_stats, load_feedback, split_dataset

examples = load_feedback("dataset/feedback.csv")
print(dataset_stats(examples))

train, val, test = split_dataset(examples)
print("train / validation / test:", len(train), len(val), len(test))
for example in train[:5]:
    print(example["label"], "|", example["comment"])