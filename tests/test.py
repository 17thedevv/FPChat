from datasets import load_dataset

dataset = load_dataset("awsaf49/persona-chat")
print(dataset["train"][0])
print(dataset)