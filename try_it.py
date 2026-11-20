from app.core import CATEGORIES, add_category_tokens, encode_example, label_token, load_gpt2

tokenizer, model = load_gpt2()
print("GPT-2 vocabulary:", len(tokenizer), "  parameters:", sum(p.numel() for p in model.parameters()))

added = add_category_tokens(tokenizer, model)
print("tokens added:", added, "  new vocabulary:", len(tokenizer))
for category in CATEGORIES:
    print(label_token(category), "-> id", tokenizer.convert_tokens_to_ids(label_token(category)))
print("embedding table:", tuple(model.get_input_embeddings().weight.shape))

example = encode_example("The cashier was so friendly and made my day.", "staff_praise", tokenizer)
print(example["input_ids"])
print(tokenizer.decode(example["input_ids"]))