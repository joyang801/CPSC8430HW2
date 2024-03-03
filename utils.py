import torch
import json
from itertools import takewhile
import matplotlib.pyplot as plt
from bleu_eval import *
import pandas as pd
import numpy as np
import re

def test_model(model, loader, device, cap_len, vocab_size, idx2word):
    print("Testing Model:")
    model.eval()
    criterion = torch.nn.NLLLoss()
    
    with torch.no_grad():
        for idx, data in enumerate(loader):
            model.zero_grad()
            features = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            preds = model(features.float(), labels)

            preds = preds.reshape(-1, cap_len-1, vocab_size)
            
            true_caps, pred_caps = detokenize(preds, data[2], idx2word)
            
            loss = 0
            for b in range(min(3, data[0].shape[0])):
                loss += criterion(preds[b,:], labels[b,1:])

                print(f'Video {b+1}:')
                print('Prediction:')
                print(pred_caps[b])
                print('Label:')
                print(true_caps[b])
            
            if idx == 0:
                break

def detokenize(preds, labels, idx2word):
    eos_tokens = ["<EOS>", "<PAD>"]
    pred_idxs = preds.max(2)[1]
    true_caps, pred_caps = [], []
    for i in range(preds.shape[0]):
        pred_cap = [idx2word[int(idx.cpu().numpy())] for idx in pred_idxs[i,:]]
        pred_cap = list(takewhile(lambda x: x not in eos_tokens, pred_cap))
        
        true_caps.append(str(labels[i]))
        pred_caps.append(" ".join(pred_cap))
    return true_caps, pred_caps

def write_results(ids, preds, labels, output_file="result.txt"):
    with open(output_file, "w") as f:
        for i in range(len(labels)):
            f.write(f"{ids[i]}, {preds[i]}\n")
            
def eval_model(mod, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename):
    print("BLEU Evaluation:")
    mod.eval()
    criterion = torch.nn.NLLLoss()
    store_labels = []
    store_predicted_labels = []
    video_labels = []
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            mod.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            # print(labels.device)
            # print(feat.device)
            predicted_labels = mod(feat.float(), labels)

            predicted_labels = predicted_labels.reshape(-1, caption_length-1, vocab_size)
            store_labels, store_predicted_labels = detoken(predicted_labels, data[2], store_labels, store_predicted_labels, detokenize_dict)
            
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
                video_labels.append(data[3][b])
    write_results(video_labels, store_predicted_labels, store_labels, output_file=output_filename)

def calc_bleu(output_file="result.txt", true_labels_path="./MLDS_hw2_1_data/testing_label.json"):
    truths = json.load(open(true_labels_path,'r'))
    results = {}
    with open(output_file,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            vid = line[:comma]
            cap = line[comma+1:]
            results[vid] = cap
    bleu_scores = []
    for item in truths:
        caps = [x.rstrip('.') for x in item['caption']]
        bleu_scores.append(BLEU(results[item['id']], caps, True))
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print("Average BLEU score is " + str(avg_bleu))

    
def plot_training_loss(losses, title='Training Loss Over Epochs', xlabel='Epoch', ylabel='Loss', save_path='/home/qiaoyiy/cpsc8430/HW2/'):

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(save_path)
    
    
class test_VideoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, test_path, test_label_path, vocab, tokens_word, tokens_idx, clength):
        super(test_VideoCaptionDataset, self).__init__()
        self.vocab = vocab
        self.tokens_word = tokens_word
        self.tokens_idx = tokens_idx
        self.clength = clength
        self.labels = pd.read_json(test_label_path)
        self._make_dataset(test_path)
        
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
        feat = self.input_feat[video_label]
        tensor_fcap = torch.Tensor(tokenized_fcap)
        one_hot = torch.nn.functional.one_hot(tensor_fcap.to(torch.int64), num_classes=len(self.tokens_idx))
        return feat, one_hot, caption, video_label

    
                
def detoken(predicted_labels, labels, store_labels, store_predicted_labels, detokenize_dict):
    endsyntax = ["<EOS>", "<PAD>"]
    predicted_labels_index = predicted_labels.max(2)[1]
    for i in range(predicted_labels.shape[0]):
        predicted_label = [detokenize_dict[int(w_idx.cpu().numpy())] for w_idx in predicted_labels_index[i,:]]
        predicted_label = list(takewhile(lambda x: x not in endsyntax, predicted_label))
        
        store_labels.append(str(labels[i]))
        store_predicted_labels.append(" ".join(predicted_label))
    return store_labels, store_predicted_labels