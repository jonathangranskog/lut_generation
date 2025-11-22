from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths = sorted(self.image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224)),
                transforms.CenterCrop((224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


if __name__ == "__main__":
    dataset = ImageDataset("data/images")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for images in dataloader:
        print(images.shape)
        break
