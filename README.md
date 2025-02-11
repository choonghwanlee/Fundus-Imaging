# Fundus-Imaging

## Get Data from Google Cloud Storage (GCS)

### Prerequisites

Before you begin, make sure you have:
1. **Google Cloud SDK** installed. If not, [download it here](https://cloud.google.com/sdk/docs/install).

### Step 1: Authenticate with Google Cloud
To authenticate your Google Cloud account, run the following command:
```bash
gcloud auth login
```

## Data Structure
- `trainLabels.csv`: CSV file containing image names and corresponding labels.
- `resized_train/`: Directory containing resized images in `.jpeg` format.

## Data Sourcing and Processing Pipeline
- **Load Labels**: Reads `trainLabels.csv` and maps labels to corresponding categories.
- **Define Dataset**: Uses `LocalImageDataset` class to load images and their labels.
- **Compute Mean and Standard Deviation**: Calculates dataset mean and standard deviation for normalization.
- **Create Dataloaders**: Splits dataset into training, validation, and test sets, then creates DataLoaders for batch processing.
- **Run the Script**: Executes the script to check the output batch shape.