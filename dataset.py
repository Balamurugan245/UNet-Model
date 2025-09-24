import os
from PIL import Image
from torch.utils.data.dataset import Dataset

class PartDrawingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)])
        self.clean = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
        self.transform = transform

    def __len__(self): return len(self.noisy)

    def __getitem__(self, idx):
      noisy = Image.open(self.noisy[idx]).convert("L")
      clean = Image.open(self.clean[idx]).convert("L")

      if self.transform:
        noisy = self.transform(noisy)
        clean = self.transform(clean)
      return noisy, clean

if __name__ == "__main__":
    root = "dataset/dataset/"
    full_dataset = PartDrawingDataset(
        os.path.join(root,"X"),
        os.path.join(root,"Y"),
        transform=None
    )
    print(len(full_dataset))
