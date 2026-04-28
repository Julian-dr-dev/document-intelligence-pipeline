
from datasets import load_dataset

dataset = load_dataset("nielsr/funsd")
print(dataset)
print(dataset["train"][0])
