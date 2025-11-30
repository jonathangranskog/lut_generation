from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

from .constants import CLIP_IMAGE_SIZE, VLM_IMAGE_SIZE


class ImageDataset(Dataset):
    def __init__(self, image_folder: str, image_size: int | None = None):
        """
        Image dataset with configurable resolution.

        Args:
            image_folder: Path to folder containing images
            image_size: Crop size. If None, uses CLIP_IMAGE_SIZE (224)
        """
        self.image_folder = image_folder
        self.image_size = image_size or CLIP_IMAGE_SIZE
        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths = sorted(self.image_paths)
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop((self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 1.0)),
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
