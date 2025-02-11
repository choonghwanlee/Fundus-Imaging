# Fundus Image Prediction for Diabetic Retinopathy

# 0. Project Overview
This project aims to develop a predictive model for detecting diabetic retinopathy (DR) from fundus images, using three distinct approaches: a basic Naive Mean model, a non-deep learning model, and a deep learning-based model. The project involves image classification across three levels of labels:

- 0: Normal
- 1: Mild Diabetic Retinopathy
- 2: Severe Diabetic Retinopathy

## Key Features
- **Model Approaches:**
    - **Naive Mean Model:** A basic baseline model that uses statistical averages for prediction.
    - **Non-Deep Learning Models:** A traditional machine learning model that doesn't rely on deep learning techniques.(**Jason fills out models u use**)
    - **Deep Learning Models:** 
        - **VGG:** A convolutional neural network (CNN) with a simple architecture, known for its deep structure and great accuracy in image classification.
        - **ResNet:** A residual network that uses skip connections to address the vanishing gradient problem and improve model performance.
        - **DenseNet:** A densely connected convolutional network that improves efficiency by connecting each layer to every other layer in a feed-forward manner.
- **Explainable AI (XAI):** To enhance model interpretability, the project integrates XAI techniques, allowing users to understand how predictions are made, fostering trust and transparency in the decision-making process.
- **Real-World Application:** A user-friendly web application where users can upload their fundus images and get real-time predictions on the severity of diabetic retinopathy.



# 1. Running Instruction
- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
- **(not sure for running)**

# 2. Data Downloading
## Prerequisites

Before you begin, make sure you have:
1. **Google Cloud SDK** installed. If not, [download it here](https://cloud.google.com/sdk/docs/install).

## Step 1: Authenticate with Google Cloud
To authenticate your Google Cloud account, run the following command:
```bash
gcloud auth login
```

## Step 2: Download Data from GCS
To download data from Google Cloud Storage, use the gsutil command. 

```bash
gsutil cp gs://aipi540-cv/classification/trainLabels.csv ./trainLabels.csv
```

```bash
gsutil cp gs://aipi540-cv/classification/resized_train/ ./resized_train/
```


## Data Structure
- `trainLabels.csv`: CSV file containing image names and corresponding labels.
- `resized_train/`: Directory containing resized images in `.jpeg` format.

# Approaches
## 1. Naive Mean Model
**(Yiqing)**

## 2. Non-Deep Learning Models
**(Jason)**

## 3. Deep Learning Models
## Data Loading
## Get Data from Google Cloud Storage (GCS)



## Data Sourcing and Processing Pipeline
- **Load Labels**: Reads `trainLabels.csv` and maps labels to corresponding categories.
- **Define Dataset**: Uses `LocalImageDataset` class to load images and their labels.
- **Compute Mean and Standard Deviation**: Calculates dataset mean and standard deviation for normalization.
- **Create Dataloaders**: Splits dataset into training, validation, and test sets, then creates DataLoaders for batch processing.
- **Run the Script**: Executes the script to check the output batch shape.

