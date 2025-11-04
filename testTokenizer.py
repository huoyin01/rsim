from riscvTokenizer import riscvTokenizer

tokenizer = riscvTokenizer()

tokens = tokenizer.tokenize("0010041b")
input_ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokens)
print(input_ids)