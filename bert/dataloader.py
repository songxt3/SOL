import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd


class SSTDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + [
            '[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


class COLADataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t', header=None,
                              names=['sentence_source', 'label', 'label_notes', 'sentence'])

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + [
            '[SEP]']  # Inserting the CLS and SEP tokens in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Pruning the list to be of the specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a PyTorch tensor

        # Obtaining the attention mask, i.e., a tensor containing 1s for non-padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


class RTEDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the premise, hypothesis, and label at the specified index in the data frame
        premise = self.df.loc[index, 'sentence1']
        hypothesis = self.df.loc[index, 'sentence2']
        label = self.df.loc[index, 'label']
        sentence = premise + '|||' + hypothesis

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the premise and hypothesis
        tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insert CLS and SEP tokens

        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Truncate to specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # Convert tokens to indices
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Convert to PyTorch tensor

        # Attention mask: 1 for non-padded tokens, 0 for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        if label == 'not_entailment':
            label = 0
        elif label == 'entailment':
            label = 1

        return tokens_ids_tensor, attn_mask, label


class WNLIDataset(Dataset):

    def __init__(self, filename, maxlen):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t', quoting=3)  # Add quoting parameter to handle quoted fields

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')  # Use the correct model name

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Selecting the sentence pairs and label at the specified index in the data frame
        sentence1 = self.df.loc[index, 'sentence1']
        sentence2 = self.df.loc[index, 'sentence2']
        label = int(self.df.loc[index, 'label'])
        sentence = sentence1 + '|||' + sentence2

        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long()
        '''
        # Tokenizing and processing the sentence
        tokens = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        tokens_ids_tensor = tokens['input_ids'].squeeze()
        attn_mask = tokens['attention_mask'].squeeze()
        '''
        return tokens_ids_tensor, attn_mask, label
