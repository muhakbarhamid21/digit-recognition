# Digit Recognition using EMNIST Dataset

I developed this digit recognition project to demonstrate an end-to-end image classification pipeline using a Convolutional Neural Network (CNN). The project uses the EMNIST Digits dataset from Kaggle and covers data loading, preprocessing, model building, training, evaluation, and saving the model in multiple formats (SavedModel, TF-Lite, and TFJS).

## Overview

I built a CNN model using TensorFlow and Keras to classify handwritten digits. The project involves:

- Downloading the EMNIST dataset directly from Kaggle.
- Preprocessing the dataset by reducing the number of images per class (to a maximum of 1,000) for balance and efficiency.
- Splitting the dataset into training, validation, and test sets.
- Building and training a CNN model with Conv2D and pooling layers.
- Evaluating the model by visualizing training and validation accuracy and loss.
- Converting and saving the trained model in three formats for deployment on different platforms.

## Dataset

The dataset used is the EMNIST Digits dataset from Kaggle:
[EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist/data)

The dataset used in this project is the EMNIST Digits dataset, which contains handwritten digit images. I manually downloaded the dataset from Kaggle and placed the CSV files in the `datasets` folder. The relevant files include:

- `emnist-digits-train.csv`
- `emnist-digits-test.csv`

I preprocess the dataset by reducing the number of images per class and then converting the images to grayscale with a resolution of 28×28 pixels. Labels are one-hot encoded.

## Model Architecture

The CNN model is built using the Keras Sequential API and includes the following layers:

- **Conv2D Layers:** For feature extraction.
- **MaxPooling2D Layers:** For downsampling.
- **Flatten Layer:** To convert the 2D feature maps into a 1D vector.
- **Dense Layers:** For classification.
- **Dropout Layer:** To reduce overfitting.

I compile the model using the Adam optimizer and categorical crossentropy as the loss function.

## Project Structure

The project is organized as follows:

```bash
.
├── README.md
├── datasets
│   ├── emnist-mnist-mapping.txt
│   ├── emnist-mnist-test.csv
│   └── emnist-mnist-train.csv
├── notebook.ipynb
├── requirements.txt
└── submission
    ├── saved_model
    │   ├── assets
    │   ├── fingerprint.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── tfjs_model
    │   ├── group1-shard1of1.bin
    │   └── model.json
    └── tflite
        ├── label.txt
        └── model.tflite
```

- **notebook.ipynb**: Contains the full code for data loading, preprocessing, model building, training, evaluation, and inference.
- **requirements.txt**: Lists all the required Python libraries for the project.
- **submission/**: Contains the trained model saved in three formats:
  - **saved_model/**: The TensorFlow SavedModel (for deployment on cloud or servers).
  - **tflite/**: The model converted to TF-Lite format (optimized for mobile and embedded devices).
  - **tfjs_model/**: The model converted to TensorFlow.js format (for running in web browsers).

## Evaluation Results

After training the model for 20 epochs, I achieved the following results:

- **Training Accuracy (last epoch):** 0.9914446648482
- **Validation Accuracy (last epoch):** 0.9829999804496765
- **Test Accuracy:** 0.9846000075340271

These metrics show that the model successfully learns to recognize handwritten digits with high accuracy on both the training and test sets, indicating strong generalization performance.

Below are the plots for accuracy and loss during training:

![plots](https://github.com/user-attachments/assets/9bb466bd-6c00-422b-8a3c-ed629e031ffe)

- The **accuracy** plot shows how both training (blue) and validation (orange) accuracy rapidly increase and stabilize above 98% by around the 10th epoch.
- The **loss** plot indicates that training loss (blue) and validation loss (orange) both decrease significantly, though validation loss slightly plateaus, suggesting the model is converging.

I evaluated the model on the test set at the end of training, which yielded a **Test Accuracy** of approximately **98.46%**. This confirms the model’s ability to generalize well to unseen data.

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/muhakbarhamid21/digit-recognition.git
   cd digit-recognition
   ```

2. **Set Up a Virtual Environment and Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**

   Manually download the EMNIST dataset files from Kaggle and place them in the datasets folder.

4. **Open and Run the Notebook:**

   Open `notebook.ipynb` in Google Colab or Jupyter Notebook and run all cells sequentially. The notebook includes:

   - Data loading and preprocessing
   - Model building and training (with EarlyStopping and ModelCheckpoint callbacks)
   - Evaluation with plots showing accuracy and loss over epochs
   - Model conversion to SavedModel, TF-Lite, and TFJS formats
   - Inference on internal test images as well as an external image

## Model Training and Evaluation

I trained the CNN model for 20 epochs with a batch size of 128. EarlyStopping and ModelCheckpoint callbacks are used to monitor validation loss and save the best model. The final model achieves an accuracy of over 85% on both training and test sets, which is verified by printed evaluation metrics and plotted graphs.

## Model Conversion

After training, I converted the model into three formats:

- SavedModel: Exported using `model.export('submission/saved_model')` (compatible with Keras 3).
- TF-Lite: Converted using `tf.lite.TFLiteConverter.from_saved_model()` and saved as `model.tflite` in the `submission/tflite` folder.
- TFJS: Converted using tensorflowjs_converter and saved in the submission/tfjs_model folder.

## Inference

The notebook demonstrates inference on an external image by:

- Loading an external image file.
- Converting it to grayscale, resizing to 28×28 pixels, normalizing, and reshaping.
- Predicting the digit using the trained model and displaying the confidence level.
