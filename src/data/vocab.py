class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.word_count = {}
        
        # Special tokens
        self.PAD_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        
        self.word2id["<pad>"] = self.PAD_ID
        self.word2id["<sos>"] = self.SOS_ID
        self.word2id["<eos>"] = self.EOS_ID
        self.word2id["<unk>"] = self.UNK_ID
        
        self.id2word[self.PAD_ID] = "<pad>"
        self.id2word[self.SOS_ID] = "<sos>"
        self.id2word[self.EOS_ID] = "<eos>"
        self.id2word[self.UNK_ID] = "<unk>"
        
        self.next_id = 4

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.next_id
            self.id2word[self.next_id] = word
            self.word_count[word] = 1
            self.next_id += 1
        else:
            self.word_count[word] = self.word_count.get(word, 0) + 1

    def build_from_text(self, texts):
        """Build vocab from list of texts."""
        from src.utils.nlp import tokenize
        
        for text in texts:
            tokens = tokenize(text)
            for token in tokens:
                self.add_word(token)

    def encode(self, text, max_len=None):
        """Convert text to token IDs."""
        from src.utils.nlp import tokenize
        
        tokens = tokenize(text)
        ids = [self.SOS_ID]
        
        for token in tokens:
            token_id = self.word2id.get(token, self.UNK_ID)
            ids.append(token_id)
        
        ids.append(self.EOS_ID)
        
        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.PAD_ID] * (max_len - len(ids))
        
        return ids

    def decode(self, ids):
        """Convert token IDs to text."""
        words = []
        for token_id in ids:
            if token_id == self.PAD_ID:
                continue
            if token_id == self.EOS_ID:
                break
            if token_id in self.id2word:
                words.append(self.id2word[token_id])
        
        return " ".join(words)

    def __len__(self):
        return self.next_id
