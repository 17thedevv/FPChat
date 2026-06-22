import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len)
        embedded = self.dropout(self.embedding(src))
        # embedded: (batch, seq_len, embedding_dim)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: (batch, seq_len, hidden_dim)
        # hidden: (n_layers, batch, hidden_dim)
        # cell: (n_layers, batch, hidden_dim)
        
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, hidden, cell, encoder_outputs):
        # input_ids: (batch, 1) - single token at a time
        # hidden: (n_layers, batch, hidden_dim)
        # cell: (n_layers, batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)
        
        embedded = self.dropout(self.embedding(input_ids))
        # embedded: (batch, 1, embedding_dim)

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        # attn_weights: (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)
        # context: (batch, 1, hidden_dim)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: (batch, 1, hidden_dim)

        logits = self.fc_out(output)
        # logits: (batch, 1, vocab_size)

        return logits, hidden, cell


class Seq2SeqChatbot(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_seq_len) - input sequences
        # tgt: (batch, tgt_seq_len) - target sequences
        # teacher_forcing_ratio: probability of using teacher forcing
        
        batch_size = tgt.shape[0]
        max_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Start with <sos> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch, 1)
        
        for t in range(1, max_len):
            logits, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            # logits: (batch, 1, vocab_size)
            
            outputs[:, t, :] = logits.squeeze(1)
            
            # Decide whether to use teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                top1 = logits.argmax(2)  # (batch, 1)
                decoder_input = top1
        
        return outputs

    def generate(self, src, max_len=20, sos_id=1, eos_id=2):
        # Generate response for a single input
        # src: (1, src_seq_len)
        
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            
            decoder_input = torch.tensor([[sos_id]]).to(self.device)
            generated = []
            
            for _ in range(max_len):
                logits, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
                # logits: (1, 1, vocab_size)
                
                top1 = logits.argmax(2)  # (1, 1)
                token_id = top1.item()
                
                if token_id == eos_id:
                    break
                
                generated.append(token_id)
                decoder_input = top1
        
        return generated
