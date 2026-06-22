import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import từ các module nội bộ của dự án
from src.data.persona_chat import load_persona_chat_pairs
from src.utils.nlp import tokenize, stem, bag_of_words
from src.models.model import NeuralNet

# ──────────────────────────────────────────────
# 1. Xác định đường dẫn động đến các file dữ liệu
# ──────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train chatbot directly on persona-chat input/output pairs."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Persona-chat split to use for training.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2000,
        help="Maximum number of persona-chat pairs to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1003,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(project_root, "checkpoints", "data.pth"),
        help="Path to save the trained checkpoint.",
    )
    return parser.parse_args()


def build_dataset(conversations):
    all_words = []
    tags = []
    xy = []

    for item in conversations:
        output = item["output"]
        tags.append(output)

        tokens = tokenize(item["input"])
        all_words.extend(tokens)
        xy.append((tokens, output))

    ignore_words = ['?', '!', '.', ',', '@', '_']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))

    unique_tags = []
    seen = set()
    for output in tags:
        if output not in seen:
            seen.add(output)
            unique_tags.append(output)

    tag_to_index = {tag: idx for idx, tag in enumerate(unique_tags)}

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tag_to_index[tag])

    return np.array(X_train), np.array(y_train), all_words, unique_tags


class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def main():
    args = parse_args()

    print(f"Loading persona-chat split={args.split}, max_examples={args.max_examples}...")
    conversations = load_persona_chat_pairs(split=args.split, max_examples=args.max_examples)
    if not conversations:
        raise RuntimeError("Không tìm thấy cặp input/output hợp lệ trong persona-chat.")

    X_train, y_train, all_words, tags = build_dataset(conversations)

    print(f"[+] Số mẫu huấn luyện : {len(X_train)}")
    print(f"[+] Kích thước từ điển : {len(all_words)}")
    print(f"[+] Số output khác nhau : {len(tags)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nBắt đầu huấn luyện mô hình chatbot...")
    for epoch in range(args.epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}")

    print(f"\nHuấn luyện hoàn tất! Loss cuối cùng: {loss.item():.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
        "persona_split": args.split,
        "num_examples": len(X_train),
    }

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(f"Đã lưu file checkpoint tại: {args.checkpoint}")


if __name__ == "__main__":
    main()
