import os

from torchvision import transforms, datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class ImageDataModule(LightningDataModule):
    
    def __init__(self, data_dir="data", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    
    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=self.transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "validation"), transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    