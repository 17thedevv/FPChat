import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.persona_chat import load_persona_chat_pairs
from src.data.vocab import Vocabulary
from src.models.seq2seq import Encoder, Decoder, Seq2SeqChatbot


def parse_args():
    parser = argparse.ArgumentParser(description="Train seq2seq chatbot on persona-chat.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--teacher-forcing", type=float, default=0.5,
                        help="Probability of using teacher forcing during training.")
    parser.add_argument("--target-loss", type=float, default=1.0,
                        help="Stop training when avg loss drops below this value.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--print-every", type=int, default=1,
                        help="Print loss every N epochs.")
    return parser.parse_args()


class Seq2SeqDataset(Dataset):
    def __init__(self, conversations, vocab, max_len):
        self.conversations = conversations
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        src_ids = self.vocab.encode(conv["input"], max_len=self.max_len)
        tgt_ids = self.vocab.encode(conv["output"], max_len=self.max_len)
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading persona-chat split={args.split}...")
    conversations = load_persona_chat_pairs(split=args.split, max_examples=args.max_examples)
    if not conversations:
        raise RuntimeError("No valid conversations found.")
    
    print(f"[+] Loaded {len(conversations)} conversation pairs")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    all_texts = []
    for conv in conversations:
        all_texts.append(conv["input"])
        all_texts.append(conv["output"])
    
    vocab.build_from_text(all_texts)
    print(f"[+] Vocabulary size: {len(vocab)}")
    
    # Create dataset and dataloader
    dataset = Seq2SeqDataset(conversations, vocab, args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    model = Seq2SeqChatbot(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            outputs = model(src, tgt, teacher_forcing_ratio=args.teacher_forcing)
            # outputs: (batch, seq_len, vocab_size)
            # tgt: (batch, seq_len)
            
            # Ignore the first decoder output because it corresponds to <sos> input,
            # and compute loss only on predictions for actual target tokens.
            output_tokens = outputs[:, 1:, :].reshape(-1, outputs.shape[-1])
            target_tokens = tgt[:, 1:].reshape(-1)
            loss = criterion(output_tokens, target_tokens)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

        if avg_loss <= args.target_loss:
            print(f"Target loss reached ({avg_loss:.4f} <= {args.target_loss}). Stopping early.")
            break
    
    print(f"\nTraining complete! Final loss: {avg_loss:.4f}")
    
    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "seq2seq.pth")
    
    torch.save({
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "vocab": vocab,
        "args": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
        }
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
