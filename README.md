# Fundus Image Prediction for Diabetic Retinopathy

# 0. Project Overview

This project aims to develop a predictive model for detecting diabetic retinopathy (DR) from fundus images, using three distinct approaches: a **basic Naive Mean model**, **non-deep learning model**, and **deep learning-based models**.

The project involves image classification across three levels of labels:

- 0: Normal
- 1: Mild Diabetic Retinopathy
- 2: Severe Diabetic Retinopathy

## Key Features

- **Model Approaches:**
  - **Naive Mean Model:** A simple baseline model that calculates the class distribution of the dataset, assuming all predictions follow this natural distribution. The model selects the most frequent class as its prediction, evaluates its performance using the F1 score.
  - **Non-Deep Learning Models:** A traditional machine learning model that doesn't rely on deep learning techniques.(**Jason fills out models u use**)
  - **Deep Learning Models:**
    - **VGG:** A convolutional neural network (CNN) with a simple architecture, known for its deep structure and great accuracy in image classification.
    - **ResNet:** A residual network that uses skip connections to address the vanishing gradient problem and improve model performance.
    - **DenseNet:** A densely connected convolutional network that improves efficiency by connecting each layer to every other layer in a feed-forward manner.
- **Explainable AI (XAI):** To enhance model interpretability, the project integrates XAI techniques, allowing users to understand how predictions are made, fostering trust and transparency in the decision-making process.
- **Real-World Application:** A user-friendly web application where users can upload their fundus images and get real-time predictions on the severity of diabetic retinopathy.

## Evaluation Metric: F1 Score

Due to the class imbalance in our dataset, we chose the F1 score as our evaluation metric. The F1 score is particularly effective in this context because it provides a balance between precision and recall, which is crucial when dealing with imbalanced data. By optimizing for F1 score, we ensure that the model is not biased toward the majority class and can effectively classify samples from all classes, providing a more reliable performance measure in the presence of data imbalance.

![Label Distribution](https://i.imghippo.com/files/xEgx2063zzk.png)

# 1. Running Instruction

- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
- **(how to run the scripts)(Jason fills out after u reformat the files)**

# 2. Data Downloading from Google Cloud Storage (GCS)

## Prerequisites

Before you begin, make sure you have
**Google Cloud SDK** installed. If not, [download it here](https://i.imghippo.com/files/xEgx2063zzk.png).

## Step 1: Authenticate with Google Cloud

To authenticate your Google Cloud account, run the following command:

```bash
gcloud auth login
```

## Step 2: Download Data from GCS

To download data from Google Cloud Storage, use the gsutil command.

```bash
gsutil -m cp -r gs://aipi540-cv/classification/trainLabels.csv ./trainLabels.csv
```

```bash
gsutil -m cp -r gs://aipi540-cv/classification/resized_train/ ./resized_train/
```

## Data Structure

- `trainLabels.csv`: CSV file containing image names and corresponding labels.
- `resized_train/`: Directory containing resized images in `.jpeg` format.

# 3. Approaches

## Naive Mean Model

The Naive Mean Model works by calculating the class distribution from the training data, where the prediction for every image in the test set is assigned to the class with the highest overall frequency.
The final F1 score is 0.6235

## Non-Deep Learning Models

For our choice of ML, the most important aspect is feature engineering & extraction. Given existing literature, we focused our image processing efforts on segmenting lesions and keypoints in a fundus image. This removes variability from different lighting and noise in teh image, allowing us to retain only the most relevant features for our model.

Our image processing pipeline consists of 8 different steps: 1. grayscaling 2. contrast enhancement 3. blurring 4. difference/subtraction 5. binary thresholding 6. noise removal 7. obtain complementary image 8. overlaying image

INSERT IMAGES HERE

After finding relevant regions, we then pass the final segmented image to a GLCM algorithm which produces Haralick Texture features (entropy, energy, contrast, etc.) as inputs to our ML classification model.

When generating features, we consider gray-level co-occurrences at 3 different distances (short, mid, long-range) to capture both local and global relationships in texture.

We use the Random Forest model for its robustness to various types of data. With hyperparameter tuning, the RF model achieves a 65.7% weighted F-1 score on the test dataset.

We attribute the lower performance (only marginally better than mean model) to the inflexibility of the image processing pipeline, which remains static despite differences in exposure, image quality, etc. between fundus images.

Future work should consider deep-learning based feature extraction pipelines that are less susceptible to variances in image quality.

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

| Model Name  | F1 Score |
| ----------- | -------- |
| ResNet18    | 0.7056   |
| VGG16       | 0.8214   |
| DenseNet121 | 0.7027   |

# 4. Explainable AI

A key motivating driver of this work is the need for explainable DR diagnoses. The use of XAI algorithms on black-box classification models allow us to explain DR severity without explicitly training a segmentation or object detection model.

The core idea is that if our classification algorithm is truly robust, it will use clinically relevant lesions such as hard/soft exudates & haemorrhages to generate predictions for mild-to-severe DR cases.

To measure this, we use the IDRiD dataset, which is a small, expert-labelled dataset of 81 fundus images and respective lesion segmentations. Specifically, we measure the IoU between ground truth segmentation and predicted Grad-CAM heatmap of our prediction.

Preliminary results indicate that our classification algorithm may learn spurious correlations during training, with just a 3% IoU with actual lesions on the IDRiD dataset.

# Application
## Demo Link
[**(DR-Detect-Model)**](https://huggingface.co/spaces/yiqing111/DR-detect)

## Run Streamlit app locally
To run the code, run the following command:
```bash
streamlit run app.py
```

Click on the Local URL (http://localhost:8501) to open the Streamlit app in your browser.

