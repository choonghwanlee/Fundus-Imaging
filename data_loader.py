import gcsfs
import pandas as pd
import torch
import io
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Define dataset information
bucket_name = "aipi540-cv"
csv_path = f"{bucket_name}/classification/trainLabels.csv"
image_folder = "classification/resized_train"

# Initialize Google Cloud Storage filesystem
fs = gcsfs.GCSFileSystem()

def load_labels():
    """Load labels from GCS."""
    with fs.open(csv_path) as f:
        df = pd.read_csv(f)
    
    # Map labels
    label_mapping = {0: "Normal", 1: "Mild Diabetic Retinopathy", 2: "Mild Diabetic Retinopathy",
                     3: "Severe Diabetic Retinopathy", 4: "Severe Diabetic Retinopathy"}
    df['category'] = df['level'].map(label_mapping)

    # Convert to numerical labels 
    category_mapping = {"Normal": 0, "Mild Diabetic Retinopathy": 1, "Severe Diabetic Retinopathy": 2}
    df['category_id'] = df['category'].map(category_mapping).fillna(-1).astype(int)

    return df

class GCSImageData(Dataset):
    def __init__(self, df, bucket_name, folder_path, transform=None):
        self.df = df
        self.bucket_name = bucket_name
        self.folder_path = folder_path
        self.transform = transform
        self.fs = gcsfs.GCSFileSystem()
        self.counter = 0

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image']
        label = int(row['category_id'])  # Ensure label is integer

        # Read image from GCS
        image_path = f"{self.bucket_name}/{self.folder_path}/{image_name}.jpeg"
        with self.fs.open(image_path, "rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        '''
        if self.counter < 10:
            print(f"Image: {image_name}, Label: {label}")
            self.counter += 1
        '''
        return image, torch.tensor(label, dtype=torch.long)

def compute_mean_std(dataset):
    """Calculate mean and std for normalization."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # Number of images in batch
        images = images.view(batch_samples, 3, -1)  # Flatten height & width

        mean += images.mean(dim=[0, 2])  # Compute mean per channel
        std += images.std(dim=[0, 2])  # Compute std per channel
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std

def get_dataloaders(batch_size=32, train_ratio=0.8, val_ratio=0.1, num_workers=4, pin_memory=False):
    """Load dataset, compute mean/std, and return DataLoaders."""
    df = load_labels()
    
    # Step 1: Load dataset without normalization
    transform_no_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = GCSImageData(df, bucket_name, image_folder, transform_no_norm)
    
    # Step 2: Compute mean & std
    mean, std = compute_mean_std(dataset)
    print(f"Computed Mean: {mean.tolist()}, Std: {std.tolist()}")

    # Step 3: Reload dataset with computed mean & std
    transform_with_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    dataset = GCSImageData(df, bucket_name, image_folder, transform_with_norm)

    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size- val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader,val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}, Labels: {labels}")
