import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer


class HateSpeechDataset(Dataset):
    """Optimized dataset for TrueHybridHateDetector model."""

    def __init__(self, jsonl_path, image_folder):
        """
        Args:
            jsonl_path: Path to JSONL file containing text, labels, and IDs
            image_folder: Base directory where images are stored
        """
        self.samples = []
        self.image_folder = image_folder

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    assert all(key in item for key in ["id", "label", "text"])
                    self.samples.append(item)
                except (json.JSONDecodeError, AssertionError):
                    continue  # Skip malformed entries

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "text": sample["text"],
            "image_path": f"{self.image_folder}/{sample['id']}.png",
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "id": sample["id"],
        }


def collate_fn(batch):
    """Batch preparation for TrueHybridHateDetector"""
    return {
        "text": [item["text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
        "ids": [item["id"] for item in batch],
    }


def create_dataloaders(config):
    """
    Creates train/val/test dataloaders.
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_set = HateSpeechDataset(config.train_dataset_path, config.image_folder_path)
    val_set = HateSpeechDataset(config.val_dataset_path, config.image_folder_path)

    loaders = [
        DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        ),
        DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        ),
    ]

    if hasattr(config, "test_dataset_path") and config.test_dataset_path:
        test_set = HateSpeechDataset(config.test_dataset_path, config.image_folder_path)
        loaders.append(
            DataLoader(
                test_set,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        )
    else:
        loaders.append(None)

    return tuple(loaders)


class Config:
    """Configuration class with default paths"""

    def __init__(self):
        self.train_dataset_path = r"K:\0505\train_70_contains_10_only.jsonl"
        self.val_dataset_path = r"K:\0505\val_15.jsonl"
        self.test_dataset_path = (
            r"K:\0505\test_15_stratified_to_50_50_757.jsonl"  # Optional  # Optional
        )
        self.image_folder_path = r"K:\thesisDataset\facebook_and_momenta"
        self.batch_size = 128


if __name__ == "__main__":
    # Debugging and verification
    print("=== Dataset and Dataloader Debugging ===")

    config = Config()
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Inspect first training batch
    first_batch = next(iter(train_loader))
    print("\nFirst training batch:")
    print(f"Texts: {first_batch['text'][:2]}")  # First 2 samples
    print(f"Image paths: {first_batch['image_paths'][:2]}")
    print(f"Labels: {first_batch['labels'][:5]}")  # First 5 labels
    print(f"Batch size: {len(first_batch['text'])}")

    # Verify image loading
    sample_image = Image.open(first_batch["image_paths"][0])
    print(f"\nImage verification: {sample_image.size} | Mode: {sample_image.mode}")
    sample_image.close()

    # Dataset statistics
    print("\nDataset stats:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
