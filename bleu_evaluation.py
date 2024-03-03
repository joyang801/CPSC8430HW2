import torch
import sys
import pandas as pd

from dataset import *
from model import *  
from utils import *
from config import *
from bleu_eval import *




# data_loc = sys.argv[1]
# output_filename = sys.argv[2]
data_loc = '/home/qiaoyiy/cpsc8430/HW2'
output_filename = 'out.txt'

test_data_path = f"{data_loc}/MLDS_hw2_1_data/testing_data/"
test_labels = str(f"{data_loc}/MLDS_hw2_1_data/testing_label.json")

train_dataset = VideoDataset(f'{data_loc}/MLDS_hw2_1_data/training_data',
                                            f'{data_loc}/MLDS_hw2_1_data/training_label.json')



batch_size = 10
device = 'cpu'

# Need to follow the setting in train.py
hidden_dim = 256
vocab_size = len(train_dataset.tokens_idx)
detokenize_dict = train_dataset.tokens_idx
feat_size = 4096
seq_length = 80
caption_length = train_dataset.clength
drop = 0.3




model = S2SModel(vocab_size, 
                         batch_size, 
                         feat_size, 
                         hidden_dim, 
                         drop, 
                         80, 
                         device, 
                         caption_length)

test_ds = test_VideoCaptionDataset(test_data_path, 
                                           test_labels, 
                                           train_dataset.vocab, 
                                           train_dataset.tokens_word, 
                                           train_dataset.tokens_idx, 
                                           train_dataset.clength)

test_dataset = torch.utils.data.DataLoader(test_ds, 
                                           batch_size=batch_size, 
                                           shuffle=True)


model.load_state_dict(torch.load("/home/qiaoyiy/cpsc8430/HW2/model.pth"))
eval_model(model, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename)
calc_bleu(output_file=output_filename, true_labels_path=test_labels)