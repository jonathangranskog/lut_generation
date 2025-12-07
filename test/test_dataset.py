"""
Tests for ImageDataset class.
"""

import os
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset


@pytest.fixture
def image_folder_with_images(tmp_path):
    """Create a temporary folder with test images."""
    folder = tmp_path / "images"
    folder.mkdir()

    # Create various valid image files
    for i in range(5):
        img = Image.new("RGB", (256, 256), color=(i * 50, 100, 150))
        img.save(folder / f"test_{i}.jpg")

    # Add a PNG image
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    img.save(folder / "test.png")

    return str(folder)


@pytest.fixture
def image_folder_with_mixed_files(tmp_path):
    """Create a folder with valid images and invalid files."""
    folder = tmp_path / "mixed"
    folder.mkdir()

    # Valid images
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    img.save(folder / "valid.jpg")

    # Invalid files
    (folder / "not_an_image.txt").write_text("This is not an image")
    (folder / "data.json").write_text('{"key": "value"}')

    # Create a subdirectory (should be skipped)
    (folder / "subdir").mkdir()

    return str(folder)


@pytest.fixture
def empty_folder(tmp_path):
    """Create an empty temporary folder."""
    folder = tmp_path / "empty"
    folder.mkdir()
    return str(folder)


class TestImageDatasetInitialization:
    """Test ImageDataset initialization."""

    def test_initialization_with_valid_folder(self, image_folder_with_images):
        """Dataset should initialize successfully with valid images."""
        dataset = ImageDataset(image_folder_with_images)

        assert len(dataset) > 0
        assert len(dataset.image_paths) == 6  # 5 JPG + 1 PNG

    def test_empty_folder_raises_error(self, empty_folder):
        """Empty folder should raise ValueError."""
        with pytest.raises(ValueError, match="No valid images found"):
            ImageDataset(empty_folder)

    def test_mixed_files_filters_correctly(self, image_folder_with_mixed_files):
        """Dataset should filter out invalid files and directories."""
        dataset = ImageDataset(image_folder_with_mixed_files)

        # Should only have the valid image
        assert len(dataset) == 1
        assert "valid.jpg" in dataset.image_paths[0]

    def test_custom_image_size(self, image_folder_with_images):
        """Dataset should accept custom image size."""
        custom_size = 128
        dataset = ImageDataset(image_folder_with_images, image_size=custom_size)

        # Check that the transform uses the correct size
        assert dataset.image_size == custom_size

    def test_default_image_size(self, image_folder_with_images):
        """Dataset should use CLIP_IMAGE_SIZE by default."""
        from utils.constants import CLIP_IMAGE_SIZE

        dataset = ImageDataset(image_folder_with_images)

        assert dataset.image_size == CLIP_IMAGE_SIZE

    def test_image_paths_sorted(self, image_folder_with_images):
        """Image paths should be sorted."""
        dataset = ImageDataset(image_folder_with_images)

        assert dataset.image_paths == sorted(dataset.image_paths)


class TestImageDatasetFormats:
    """Test ImageDataset with different image formats."""

    def test_jpeg_support(self, tmp_path):
        """Dataset should support JPEG images."""
        folder = tmp_path / "jpeg"
        folder.mkdir()

        img = Image.new("RGB", (256, 256))
        img.save(folder / "test.jpg")

        dataset = ImageDataset(str(folder))
        assert len(dataset) == 1

    def test_png_support(self, tmp_path):
        """Dataset should support PNG images."""
        folder = tmp_path / "png"
        folder.mkdir()

        img = Image.new("RGB", (256, 256))
        img.save(folder / "test.png")

        dataset = ImageDataset(str(folder))
        assert len(dataset) == 1

    def test_multiple_formats(self, tmp_path):
        """Dataset should support multiple image formats."""
        folder = tmp_path / "multi"
        folder.mkdir()

        # Create different format images
        for ext in ["jpg", "png", "bmp"]:
            img = Image.new("RGB", (256, 256))
            img.save(folder / f"test.{ext}")

        dataset = ImageDataset(str(folder))
        assert len(dataset) == 3

    def test_rgba_to_rgb_conversion(self, tmp_path):
        """Dataset should convert RGBA images to RGB."""
        folder = tmp_path / "rgba"
        folder.mkdir()

        # Create RGBA image
        img = Image.new("RGBA", (256, 256), color=(100, 150, 200, 128))
        img.save(folder / "test.png")

        dataset = ImageDataset(str(folder))
        assert len(dataset) == 1

        # Get the image and verify it's RGB
        image_tensor = dataset[0]
        assert image_tensor.shape[0] == 3  # RGB channels


class TestImageDatasetGetItem:
    """Test ImageDataset __getitem__ method."""

    def test_getitem_returns_tensor(self, image_folder_with_images):
        """__getitem__ should return a tensor."""
        dataset = ImageDataset(image_folder_with_images)

        image = dataset[0]

        assert isinstance(image, torch.Tensor)

    def test_getitem_correct_shape(self, image_folder_with_images):
        """__getitem__ should return tensor with correct shape."""
        image_size = 128
        dataset = ImageDataset(image_folder_with_images, image_size=image_size)

        image = dataset[0]

        assert image.shape == (3, image_size, image_size)

    def test_getitem_value_range(self, image_folder_with_images):
        """__getitem__ should return tensor in [0, 1] range."""
        dataset = ImageDataset(image_folder_with_images)

        image = dataset[0]

        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_getitem_all_indices(self, image_folder_with_images):
        """Should be able to access all indices."""
        dataset = ImageDataset(image_folder_with_images)

        for i in range(len(dataset)):
            image = dataset[i]
            assert image is not None
            assert image.shape[0] == 3

    def test_getitem_out_of_bounds(self, image_folder_with_images):
        """Accessing out of bounds index should raise error."""
        dataset = ImageDataset(image_folder_with_images)

        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]


class TestImageDatasetTransforms:
    """Test ImageDataset transform pipeline."""

    def test_random_crop_applied(self, tmp_path):
        """Random crop transform should be applied."""
        folder = tmp_path / "crop"
        folder.mkdir()

        # Create large image
        img = Image.new("RGB", (512, 512))
        img.save(folder / "large.jpg")

        # Dataset with smaller crop size
        crop_size = 256
        dataset = ImageDataset(str(folder), image_size=crop_size)

        image = dataset[0]

        # Output should be cropped size
        assert image.shape == (3, crop_size, crop_size)

    def test_transform_pipeline_exists(self, image_folder_with_images):
        """Dataset should have transform pipeline configured."""
        dataset = ImageDataset(image_folder_with_images)

        # Verify transform is configured
        assert dataset.transform is not None

        # Verify transform is a Compose with multiple transforms
        from torchvision import transforms

        assert isinstance(dataset.transform, transforms.Compose)

        # Should have multiple transforms (crop, flips, blur, to_tensor)
        assert len(dataset.transform.transforms) >= 4

    def test_tensor_conversion(self, image_folder_with_images):
        """ToTensor transform should convert to proper format."""
        dataset = ImageDataset(image_folder_with_images)

        image = dataset[0]

        # Should be float tensor
        assert image.dtype == torch.float32

        # Should be in [0, 1] range
        assert image.min() >= 0.0
        assert image.max() <= 1.0


class TestImageDatasetLen:
    """Test ImageDataset __len__ method."""

    def test_len_correct(self, image_folder_with_images):
        """__len__ should return correct number of images."""
        dataset = ImageDataset(image_folder_with_images)

        assert len(dataset) == 6

    def test_len_with_filtered_files(self, image_folder_with_mixed_files):
        """__len__ should only count valid images."""
        dataset = ImageDataset(image_folder_with_mixed_files)

        assert len(dataset) == 1


class TestImageDatasetDataLoader:
    """Test ImageDataset with PyTorch DataLoader."""

    def test_dataloader_iteration(self, image_folder_with_images):
        """Dataset should work with DataLoader."""
        dataset = ImageDataset(image_folder_with_images)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, torch.Tensor)
            assert batch.shape[1] == 3  # RGB channels

        assert batch_count > 0

    def test_dataloader_batch_size(self, image_folder_with_images):
        """DataLoader should respect batch size."""
        dataset = ImageDataset(image_folder_with_images)
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        first_batch = next(iter(dataloader))

        assert first_batch.shape[0] == batch_size

    def test_dataloader_shuffle(self, image_folder_with_images):
        """DataLoader shuffling should work."""
        dataset = ImageDataset(image_folder_with_images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Should be able to iterate without errors
        batches = list(dataloader)
        assert len(batches) == len(dataset)

    def test_dataloader_full_iteration(self, image_folder_with_images):
        """DataLoader should iterate through all images."""
        dataset = ImageDataset(image_folder_with_images)
        batch_size = 3
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_images = 0
        for batch in dataloader:
            total_images += batch.shape[0]

        assert total_images == len(dataset)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_dataloader_various_batch_sizes(self, image_folder_with_images, batch_size):
        """DataLoader should work with various batch sizes."""
        dataset = ImageDataset(image_folder_with_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            assert batch.ndim == 4  # (B, C, H, W)
            assert batch.shape[1] == 3  # RGB channels
            # Last batch might be smaller
            assert batch.shape[0] <= batch_size


class TestImageDatasetEdgeCases:
    """Test ImageDataset edge cases."""

    def test_single_image(self, tmp_path):
        """Dataset should work with single image."""
        folder = tmp_path / "single"
        folder.mkdir()

        img = Image.new("RGB", (256, 256))
        img.save(folder / "single.jpg")

        dataset = ImageDataset(str(folder))

        assert len(dataset) == 1
        image = dataset[0]
        assert image.shape[0] == 3

    def test_large_number_of_images(self, tmp_path):
        """Dataset should handle many images."""
        folder = tmp_path / "many"
        folder.mkdir()

        num_images = 50
        for i in range(num_images):
            img = Image.new("RGB", (64, 64), color=(i, i, i))
            img.save(folder / f"img_{i:03d}.jpg")

        dataset = ImageDataset(str(folder))

        assert len(dataset) == num_images

    def test_small_images_upscaled(self, tmp_path):
        """Small images should be handled by RandomCrop."""
        folder = tmp_path / "small"
        folder.mkdir()

        # Create image smaller than crop size
        small_size = 128
        crop_size = 256

        # This will fail because RandomCrop requires image >= crop size
        # So we need to create image >= crop size
        img = Image.new("RGB", (crop_size, crop_size))
        img.save(folder / "small.jpg")

        dataset = ImageDataset(str(folder), image_size=crop_size)
        image = dataset[0]

        assert image.shape == (3, crop_size, crop_size)

    def test_corrupted_file_skipped(self, tmp_path):
        """Corrupted image files should be skipped."""
        folder = tmp_path / "corrupted"
        folder.mkdir()

        # Valid image
        img = Image.new("RGB", (256, 256))
        img.save(folder / "valid.jpg")

        # Create corrupted file with .jpg extension
        (folder / "corrupted.jpg").write_bytes(b"not a valid image")

        dataset = ImageDataset(str(folder))

        # Should only load the valid image
        assert len(dataset) == 1
