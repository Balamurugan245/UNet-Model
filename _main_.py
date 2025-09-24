import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from dataset import PartDrawingDataset
from unet import unet
from loss_function import IoULoss
import torch.nn.functional as F

if __name__ == "__main__":
      EPOCHS = 30
      lr=1e-4
      batch_size=8
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      root = "dataset/dataset/"
      full_dataset = PartDrawingDataset(
          os.path.join(root,"X"),
          os.path.join(root,"Y"),
          transform=None
      )
      train_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.ToTensor(),
      ])
    
        val_transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
        ])
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        
        train_set.dataset.transform = train_transform
        val_set.dataset.transform = val_transform
        
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size)
        
        model = unet( in_channel=1, num_classes=1).to(device)
        criterion = IoULoss()
        optimizer = optim.Adam(model.parameters(),lr)
        scaler = torch.amp.GradScaler()
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                noisy = noisy.to(device)
                clean = clean.to(device).float()
        
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(noisy)
                    loss = criterion(outputs, clean)
        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
                train_loss += loss.item()
            print(f"Epoch {epoch+1}, Train Loss={train_loss/len(train_loader):.4f}")
        
        def show_test_predictions():
            model.eval()
            noisy_imgs, clean_imgs = next(iter(val_loader))
            num_images_to_show = min(5, noisy_imgs.size(0))
            noisy_imgs = noisy_imgs[:num_images_to_show].to(device)
            clean_imgs = clean_imgs[:num_images_to_show]
        
            with torch.no_grad():
                outputs = model(noisy_imgs)
                outputs = torch.sigmoid(outputs)
                outputs = 1 - outputs
                outputs = outputs.cpu()
        
            noisy_imgs = noisy_imgs.cpu()
        
            fig, axes = plt.subplots(num_images_to_show, 3, figsize=(12, 2 * num_images_to_show))
            if num_images_to_show == 1:
                axes = axes.reshape(1, -1)
        
            for i in range(num_images_to_show):
                axes[i, 0].imshow(noisy_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[i, 0].set_title("Noisy")
                axes[i, 1].imshow(outputs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title("Predicted")
                axes[i, 2].imshow(clean_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[i, 2].set_title("Clean")
        
                for j in range(3):
                    axes[i, j].axis("off")
        
            plt.tight_layout()
            plt.show()
        
        show_test_predictions()
