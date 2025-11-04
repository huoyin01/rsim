import os

from transformers import PreTrainedTokenizer
from transformers.utils import logging

from utils import INSTRUCTION_MAX_LENGTH


logger = logging.get_logger(__name__)


class riscvTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = self.get_vocab()
        self.add_special_tokens({'pad_token': '<pad>'})
        # self._pad_token_type_id = 2
        
    @property
    def vocab_size(self):
        return len(self.vocab)    
    
    def get_vocab(self):
        vocab = {}
        for i in range(256):
            vocab[str(i)] = i
        vocab['<pad>'] = 256
        return vocab    
    
    def _tokenize(self, text, **kwargs):
        # tokens = []
        # for i in range(0, len(text), 2):
        #     tokens.append(text[i:i+2])
        tokens = text.split(',')
        # if len(tokens) < INSTRUCTION_MAX_LENGTH:
        #     tokens.extend(['<pad>'] * (INSTRUCTION_MAX_LENGTH - len(tokens)))
        return tokens
    
    def _convert_token_to_id(self, token):
        return self.vocab[token]
    
    def _convert_id_to_token(self, id):
        if id == 256:
            return '<pad>'
        return str(id)
    
    def convert_tokens_to_string(self, tokens):
        return ",".join(tokens).strip()
    
    def decode(self, ids):
        tokens = []
        for id in ids:
            tokens.append(self._convert_id_to_token(id.item()))
        return self.convert_tokens_to_string(tokens)
    
    def save_vocabulary(self, save_directory, filename_prefix = None):
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    
    