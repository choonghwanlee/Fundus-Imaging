# Fundus Image Prediction for Diabetic Retinopathy

# 0. Project Overview
This project aims to develop a predictive model for detecting diabetic retinopathy (DR) from fundus images, using three distinct approaches: a **basic Naive Mean model**, **non-deep learning model**, and **deep learning-based models**. 

The project involves image classification across three levels of labels:
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

## Evaluation Metric: F1 Score
Due to the class imbalance in our dataset, we chose the F1 score as our evaluation metric. The F1 score is particularly effective in this context because it provides a balance between precision and recall, which is crucial when dealing with imbalanced data. By optimizing for F1 score, we ensure that the model is not biased toward the majority class and can effectively classify samples from all classes, providing a more reliable performance measure in the presence of data imbalance.



# 1. Running Instruction
- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
- **(not sure for running)**

# 2. Data Downloading from Google Cloud Storage (GCS)
## Prerequisites
Before you begin, make sure you have
**Google Cloud SDK** installed. If not, [download it here](https://cloud.google.com/sdk/docs/install).

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
## Naive Mean Model
**(Yiqing)**

## Non-Deep Learning Models
**(Jason)**

## Deep Learning Models

## Data Sourcing and Processing Pipeline
- **Load Labels**: Reads `trainLabels.csv` and maps labels to corresponding categories.
- **Define Dataset**: Uses `LocalImageDataset` class to load images and match them with their labels.
- **Compute Mean and Standard Deviation**: Calculates dataset mean and standard deviation for normalization.
- **Create Dataloaders**: Splits dataset into training, validation, and test sets, then creates DataLoaders for batch processing.
- **Run the Script**: Executes the script to check the output batch shape.
  
## Data Preprocessing
To prevent overfitting and ensure better generalization, the following techniques were applied:
- **Oversampling** is applied to balance class distribution during training.
- **Batch Normalization** and **Dropout** are used to prevent overfitting
- **Early Stopping** is implemented to halt training when validation loss doesn't improve for a specified number of epochs.
- **Adam Optimizer** with learning rate of 0.0001 and weight decay (1e-5) is used for optimization.

## Results 
| Model Name     | F1 Score     |
|----------------|--------------|
| ResNet18       | 0.7056       |
| VGG16          | 0.8214       |
| DenseNet121    | 0.7027       |

# Explainable AI
**(Jason)**

# Application
**(Yiqing pastes the link)**


