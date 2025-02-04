import gcsfs
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io

# Initialize Google Cloud Storage filesystem
fs = gcsfs.GCSFileSystem()

class GCSDataset(Dataset):
    def __init__(self, bucket_name, folder_path, transform=None):
        self.bucket_name = bucket_name
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = fs.ls(f"{bucket_name}/{folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with fs.open(img_path, "rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define dataset
bucket_name = "aipi540-cv"
folder_path = "classification/resized_train/"
dataset = GCSDataset(bucket_name, folder_path, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example: Load a batch
images = next(iter(dataloader))
print(images.shape)  # Should print (batch_size, 3, 224, 224)
