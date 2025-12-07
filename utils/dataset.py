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

        # Get all files and filter by checking if PIL can identify the format
        # This supports all PIL-supported formats (JPEG, PNG, BMP, GIF, TIFF, WebP, etc.)
        self.image_paths = []
        for filename in os.listdir(image_folder):
            filepath = os.path.join(image_folder, filename)
            # Skip directories
            if os.path.isdir(filepath):
                continue
            # Check if PIL recognizes this as an image format (without fully loading it)
            try:
                with Image.open(filepath) as img:
                    img.format  # Access format to verify PIL can read it
                self.image_paths.append(filepath)
            except Exception:
                # Not a recognized image format, skip silently
                continue

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
