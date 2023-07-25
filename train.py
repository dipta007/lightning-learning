from dataset import TwitterDataModule
from model import TwitterModelModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

callbacks = [EarlyStopping(monitor="val_loss", mode="min")]

datamodule = TwitterDataModule()
model = TwitterModelModule()
trainer = pl.Trainer(callbacks=callbacks, accelerator='mps', devices=1)

trainer.fit(model=model, datamodule=datamodule)
