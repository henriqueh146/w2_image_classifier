import yaml
import os

import pytorch_lightning as pl

from models.model import BinaryImageClassifier
from data.datamodule import ImageDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    datamodule = ImageDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    model = BinaryImageClassifier(lr=config["lr"])

    logger = TensorBoardLogger("tb_logs", name="horse_human_classifier")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",
        filename="horse_human_classifier",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("models/horse_human_classifier.ckpt")


if __name__ == "__main__":
    main()
