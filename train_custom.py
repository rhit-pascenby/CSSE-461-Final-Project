import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F_tv
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import sys
from functools import partial
from typing import Dict, List, Tuple
import datasets  # Import the datasets library

# Add the MiDaS-master directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import MiDaS components
from midas.model_loader import load_model

# Import torch.nn.functional for F.interpolate
import torch.nn.functional as F  # <--- NEW IMPORT

# --- Configuration ---
MODEL_TYPE = "dpt_large_384"
PRETRAINED_WEIGHTS_PATH = "weights/dpt_large_384.pt"

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20  # Reduced epochs for faster debugging
SAVE_MODEL_PATH = "finetuned_midas_nyuv2.pt"
DATA_DIR = "data/nyuv2"  # Define a data directory.  This might not be needed with HF Datasets.

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Pre-trained MiDaS Model ---
print("Loading MiDaS model...")
model, midas_transform_inference, net_w, net_h = load_model(
    device,
    PRETRAINED_WEIGHTS_PATH,
    MODEL_TYPE,
    optimize=False,
    height=None, square=False
)
model.train()
print(f"Model loaded. Expected input size: {net_w}x{net_h}")


# --- 2. Load NYUv2 Dataset using Hugging Face Datasets ---
def load_nyuv2_dataset():
    """Loads the NYUv2 dataset using Hugging Face Datasets."""
    try:
        dataset = datasets.load_dataset('tanganke/nyuv2')
        dataset = dataset.with_format('torch')  # Convert to torch.Tensor
        # Merge the train and validation splits, handling potential missing 'validation' split
        if "validation" in dataset:
            full_dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"]])
        elif "val" in dataset:  # Check for "val" as an alternative
            full_dataset = datasets.concatenate_datasets([dataset["train"], dataset["val"]])
        else:
            full_dataset = dataset["train"]  # Only use the training split if validation is not available.
            print("Warning: 'validation' split not found. Using only the 'train' split.")
        return full_dataset
    except Exception as e:
        print(f"Error loading NYUv2 dataset from Hugging Face: {e}")
        print(
            "Please ensure you have the 'datasets' library installed (`pip install datasets`) "
            "and that you have a stable internet connection."
        )
        sys.exit(1)



# --- NEW: Combined Preprocessing and Resize Function ---
class PreprocessAndResize(nn.Module):  # Make it a PyTorch module
    def __init__(self, size=(384, 384)):
        super().__init__()
        self.size = size
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(self, batch):
        #  input is a dictionary
        img = batch["image"]
        depth = batch["depth"]

        # Add batch dimension for interpolation if it doesn't exist
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)

        # Resize using F.interpolate
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=self.size, mode='nearest')

        # Normalize image tensor
        img = self.normalize(img)

        # Calculate inverted depth and mask
        valid_mask = (depth > 0).float()
        inverted_depth = torch.zeros_like(depth)
        inverted_depth[valid_mask.bool()] = 1.0 / (depth[valid_mask.bool()] + 1e-6)

        return img, inverted_depth, valid_mask.squeeze(0)  # Return mask


def make_dataloaders(full_dataset, batch_size, test_size, num_workers=0):
    """
    Creates training and testing DataLoaders for the NYUv2 dataset.

    Args:
        full_dataset (Dataset): The full NYUv2 dataset.
        batch_size (int): Batch size for the DataLoaders.
        test_size (float): Proportion of the dataset to use for testing.
        num_workers (int, optional): Number of workers for the DataLoaders. Defaults to 0.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # 1. Create the combined transform
    transform = PreprocessAndResize(size=(384, 384))

    # 2. Split the dataset into training and testing sets.
    num_samples = len(full_dataset)
    test_len = int(num_samples * test_size)
    train_len = num_samples - test_len
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_len, test_len])

    # 3. Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader



# --- 4. Create Data Loaders ---
test_size = 0.2  # 20% of data will be for testing
batch_size = BATCH_SIZE
full_dataset = load_nyuv2_dataset() # Load the dataset
train_loader, test_loader = make_dataloaders(
    full_dataset, batch_size, test_size
)
transform = PreprocessAndResize(size=(384, 384)) #define transform here

# --- 5. Define Loss Function and Optimizer ---
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 6. Train the Model ---
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)") as t_train_loader:
            for batch in t_train_loader:  # Unpack directly
                image, label, mask = transform(batch) # Preprocess the batch
                image, label, mask = image.to(device), label.to(device), mask.to(device)

                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs * mask, label * mask)  # Apply mask to loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                t_train_loader.set_postfix(loss=loss.item())
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)") as t_test_loader:
            with torch.no_grad():
                for batch in t_test_loader:
                    image, label, mask = transform(batch) # Preprocess the batch
                    image, label, mask = image.to(device), label.to(device), mask.to(device)
                    outputs = model(image)
                    loss = criterion(outputs * mask, label * mask)
                    total_val_loss += loss.item()
                    t_test_loader.set_postfix(loss=loss.item())
        avg_val_loss = total_val_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Trained model saved to {SAVE_MODEL_PATH}")



# --- 7. Run Training ---
if __name__ == "__main__":
    train_model(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, device)
