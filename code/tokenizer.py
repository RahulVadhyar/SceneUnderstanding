import pickle
from collections import defaultdict
def returnunk():
    return "[UNK]"
class Tokenizer:
    def __init__(self):
        self.char_to_idx = defaultdict(returnunk)
        self.idx_to_char = defaultdict(returnunk)
        self.vocab_size = 0
    
    def train(self, text):
        chars = list(set(text))
        self.char_to_idx = {ch:i for i, ch in enumerate(chars)}
        self.idx_to_char = {i:ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        output = []
        num_unk = 0
        output.append(self.char_to_idx["[START]"])
        if text == '[UNK]':
            return self.char_to_idx["[UNK]"]
            
        for ch in text:
            if ch not in self.char_to_idx:
                ch = self.char_to_idx["[UNK]"]
                num_unk += 1
            else:
                ch = self.char_to_idx[ch]
            output.append(ch)
        output.append(self.char_to_idx["[END]"])

        if num_unk > 0.1 * len(text):
            return []
        return output
    
    def decode(self, encoded):
        string = ""
        for idx in encoded:
            string += self.idx_to_char[idx]

        return string
    

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    def load(self, filepath):
        with open(filepath, "rb") as f:
            tokenizer = pickle.load(f)
        self.char_to_idx = tokenizer.char_to_idx
        self.idx_to_char = tokenizer.idx_to_char
        self.vocab_size = tokenizer.vocab_size
        return self

    def get_vocab_size(self):
        return self.vocab_size