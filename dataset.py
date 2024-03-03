import torch
import re
import numpy as np
from collections import Counter, OrderedDict
import pandas as pd


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, label_path, is_train=True, vocab=None, tokens_word=None, tokens_idx=None, clength=None):
        super(VideoDataset, self).__init__()
        self.is_train = is_train
        self.labels = pd.read_json(label_path)

        if is_train:
            self._build_vocab()
            self._build_word_index()
        else:
            self.vocab = vocab
            self.tokens_word = tokens_word
            self.tokens_idx = tokens_idx
            self.clength = clength
        
        self._make_dataset(input_path)
        
    def _build_vocab(self):
        self.vocab = Counter()
        self.clength = 0
        for row in self.labels.itertuples(index=False):
            for fcaps in np.unique(np.array(row[0])):
                fcap_adj = re.sub(r'[^\w\s]', '', fcaps).lower().split(" ")
                self.vocab += Counter(fcap_adj)
                self.clength = len(fcap_adj) if len(fcap_adj) > self.clength else self.clength
        self.vocab = Counter(dict(filter(lambda i: i[1] > 3, self.vocab.items())))
        self.clength += 2
    
    def _build_word_index(self):
        # PAD is padding, BOS begin sentence, EOS end sentence, UNK unknown word
        self.tokens_idx = {0: "<PAD>", 1:"<BOS>", 2:"<EOS>", 3: "<UNK>"}
        self.tokens_word = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for idx, w in enumerate(self.vocab):
            self.tokens_idx[idx+4] = w
            self.tokens_word[w] = idx+4
    
    def _make_dataset(self, training_path):
        self.input_feat = {}
        self.output_fcap_info = []
        for row in self.labels.itertuples(index=False):
            self.input_feat[row[1]] = torch.from_numpy(np.load(f"{training_path}/feat/{row[1]}.npy"))
            for fcaps in np.unique(np.array(row[0])):
                full_fcap = ["<BOS>"]
                for w in re.sub(r'[^\w\s]', '', fcaps).lower().split(" "):
                    if w in self.vocab:
                        full_fcap.append(w)
                    else:
                        full_fcap.append("<UNK>")
                full_fcap.append("<EOS>")
                full_fcap.extend(["<PAD>"]*(self.clength-len(full_fcap)))
                tokenized_fcap = [self.tokens_word[word] for word in full_fcap]
                
                # Index 0: Caption, Index 1: Full Cap w/ string info, Index 2: OneHotEncoded, Index 3: Video Label
                self.output_fcap_info.append([fcaps, full_fcap, tokenized_fcap, row[1]])
                
    def __len__(self):
        return len(self.output_fcap_info)
    
    def __getitem__(self, idx):
        caption, _, tokenized_fcap, video_label = self.output_fcap_info[idx]
        tensor_fcap = torch.Tensor(tokenized_fcap)
        one_hot = torch.nn.functional.one_hot(tensor_fcap.to(torch.int64), num_classes=len(self.tokens_idx))
        return self.input_feat[video_label], one_hot, caption
