import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MRPCDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t', quoting=3)  # Add quoting parameter to handle quoted fields

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence pairs and label at the specified index in the data frame
        sentence1 = self.df.loc[index, '#1 String']
        sentence2 = self.df.loc[index, '#2 String']
        label = int(self.df.loc[index, 'Quality'])
        sentence = sentence1 + '|||' + sentence2

        # Tokenizing and processing the sentence
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

