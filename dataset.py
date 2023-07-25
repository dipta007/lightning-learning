import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
from transformers import AutoTokenizer
import random

class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, file_dir='./data/train.csv', model_name='facebook/bart-large', batch_size=128) -> None:
        super().__init__()
        self.file_dir = file_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        df = pd.read_csv(self.file_dir)
        self.data = df['text'].tolist()
        random.shuffle(self.data)
        
        total = len(self.data)
        train_index, val_index = int(total*0.7), int(total*0.7) + int(total*0.2)
        self.train_data = self.data[:train_index]
        self.val_data = self.data[train_index:val_index]
        self.test_data = self.data[train_index+val_index:]

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_data = self.tokenizer(self.train_data, return_tensors='pt', padding='longest')
            self.train_data = [{'input_ids': self.train_data['input_ids'][i], 'attention_mask': self.train_data['attention_mask'][i]} for i in range(len(self.train_data['input_ids']))]
            
            self.val_data = self.tokenizer(self.val_data, return_tensors='pt', padding='longest')
            self.val_data = [{'input_ids': self.val_data['input_ids'][i], 'attention_mask': self.val_data['attention_mask'][i]} for i in range(len(self.val_data['input_ids']))]
        if stage == 'test':
            self.test_data = self.tokenizer(self.test_data, return_tensors='pt', padding='longest')
            self.test_data = [{'input_ids': self.test_data['input_ids'][i], 'attention_mask': self.test_data['attention_mask'][i]} for i in range(len(self.test_data['input_ids']))]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)



