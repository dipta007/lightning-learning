import lightning.pytorch as pl
from transformers import BartForConditionalGeneration
import torch

class TwitterModelModule(pl.LightningModule):
    def __init__(self, model_name='facebook/bart-large') -> None:
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['input_ids'].size()[0]
        seq_len = batch['input_ids'].size()[1]

        y = batch['input_ids']
        y_hat = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        y_hat = y_hat.reshape(batch_size*seq_len, -1)
        y = y.reshape(-1)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        batch_size = batch['input_ids'].size()[0]
        seq_len = batch['input_ids'].size()[1]

        y = batch['input_ids']
        y_hat = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        y_hat = y_hat.reshape(batch_size*seq_len, -1)
        y = y.reshape(-1)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
