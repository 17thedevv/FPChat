import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.seq2seq import Encoder, Decoder, Seq2SeqChatbot


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found at {checkpoint_path}")
        print("    Please run: python -m src.training.train_seq2seq")
        exit()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    args = checkpoint["args"]
    
    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        n_layers=args["n_layers"],
        dropout=args["dropout"]
    ).to(device)
    
    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        n_layers=args["n_layers"],
        dropout=args["dropout"]
    ).to(device)
    
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])
    
    model = Seq2SeqChatbot(encoder, decoder, device)
    
    return model, vocab


def generate_response(model, vocab, user_input, device, max_len=30):
    """Generate response using seq2seq model."""
    model.eval()
    
    # Encode user input
    src_ids = vocab.encode(user_input, max_len=50)
    src_tensor = torch.tensor([src_ids]).to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(src_tensor, max_len=max_len, 
                                      sos_id=vocab.SOS_ID, 
                                      eos_id=vocab.EOS_ID)
    
    # Decode to text
    response = vocab.decode(generated_ids)
    return response


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("checkpoints", "seq2seq.pth")
    
    print("Loading model...")
    model, vocab = load_checkpoint(checkpoint_path, device)
    
    bot_name = "FPBot"
    print("=" * 50)
    print(f"{bot_name}: Hello! I'm ready to chat. (Type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        try:
            response = generate_response(model, vocab, user_input, device, max_len=30)
            print(f"{bot_name}: {response}")
        except Exception as e:
            print(f"{bot_name}: Sorry, I had an error: {e}")


if __name__ == "__main__":
    main()
