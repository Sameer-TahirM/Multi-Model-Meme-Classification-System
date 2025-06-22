# Multi-Model Meme Sentiment Classification


> A multi-model machine learning system that classifies meme sentiment using an ensemble of image and text-based models, with a user-friendly Flask web interface for predictions.

## Description

This project implements a multi-model sentiment classification system for internet memes. The goal is to determine whether a meme is positive, negative, or neutral by analyzing both its visual content and the embedded text. This is achieved by training an ensemble of six different machine learning models—three for image classification and three for text classification—and combining their predictions through a majority vote.

The project leverages `scikit-learn` for model training, `scikit-image` and `Pillow` for image preprocessing, and `easyocr` for text extraction. Finally, the trained models are deployed in a user-friendly **Flask web application**, allowing users to upload a meme and receive a sentiment prediction.

## How It Works

The system uses a two-pronged approach to classify a meme:

1.  **Image Analysis**: The image is preprocessed by converting it to grayscale, resizing it to a standard 200x200 dimension, and flattening it into a feature vector. This vector is then fed into three distinct image-based classifiers.
2.  **Text Analysis**: **Optical Character Recognition (OCR)** using `easyocr` extracts text from the image. The text is cleaned, lemmatized, and converted into a numerical vector using **TF-IDF**. This vector is then classified by three separate text-based models.
3.  **Ensemble Prediction**: The predictions from all six models are collected. The final sentiment is determined by a **majority vote** (the mode) of these predictions, providing a robust, multi-modal classification.

## Features

- **Multi-Modal Pipeline**: Combines both computer vision and NLP techniques for a more accurate sentiment analysis.
- **Image Feature Extraction**: Preprocessing pipeline includes resizing, grayscaling, and flattening image data.
- **Text Feature Extraction**: Utilizes OCR to extract text, followed by cleaning, lemmatization, and TF-IDF vectorization.
- **Ensemble Learning**: Employs a majority voting system across six different classifiers for the final prediction.
- **Classic ML Models**: Implements Random Forest, KNN, Decision Tree, Logistic Regression, Naive Bayes, and SVM.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the optimal parameters for each model.
- **Model Persistence**: Trained models are saved using `pickle` for easy deployment.
- **Web Application**: A Flask-based web interface for easy user interaction and real-time predictions.

## Model Pipeline & Performance

The system uses two sets of three classifiers, one for each modality.

#### Image Classifiers:
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

#### Text Classifiers:
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

### Performance
The models were trained and evaluated on the dataset. The table below shows the F1-Score (macro average) for each model on the test set.

| Model                       | Modality | F1-Score (Macro) |
| --------------------------- | :------: | :--------------: |
| **KNN**                     |  Image   |      ~33.3%      |
| **Decision Tree**           |  Image   |      ~32.6%      |
| **Random Forest**           |  Image   |      ~28.9%      |
| **Logistic Regression**     |   Text   |      ~34.3%      |
| **Naive Bayes**             |   Text   |      ~33.5%      |
| **Support Vector Machine**  |   Text   |      ~30.8%      |

*(Note: The final deployed system uses majority voting on all six models to produce a more robust prediction.)*


## Usage

1.  **Data Preprocessing and Training**:
    - Run the `ModelTraining+DataPreprocessing.ipynb` notebook.
    - This will load the images and labels, preprocess the data, train all six models using `GridSearchCV`, and save the trained models as `.pkl` files.
    - **Note**: Ensure the `images` folder and `labels.csv` are in the correct path as specified in the notebook.

2.  **Making a Prediction**:
    - The `Prediction_testing.ipynb` notebook demonstrates how to make a prediction on a single meme.
    - Set the `image_path` variable to your test image and run the cells.
    - The notebook will load the six saved models and output the final sentiment based on a majority vote.

3.  **Running the Flask Web App** (if applicable):
    - To run the web application for predictions:
    ```bash
    flask run
    # or
    python app.py
    ```
    - Open your browser and navigate to `http://127.0.0.1:5000` to upload a meme and see the classification.

## Dependencies

The project requires the following libraries. You can install them all using `pip install -r requirements.txt`.

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `Pillow`
- `nltk`
- `easyocr`
- `opencv-python`
- `Flask` (for web deployment)

## Dataset (Local)
- **Description**: The dataset contains thousands of memes with associated text and overall sentiment annotations, categorized as `positive`, `negative`, `neutral`, `very_positive`, and `very_negative`. These are mapped to three classes (positive, negative, neutral) for this project.
