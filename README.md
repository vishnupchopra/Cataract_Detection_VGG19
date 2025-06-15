# 👁️ Cataract Detection using VGG19

This repository implements a deep learning approach for cataract detection in retinal fundus images using the VGG19 convolutional neural network architecture.

## 📋 Table of Contents

- [🔍 Overview](#-overview)
- [📚 Dataset](#-dataset)
- [⚙️ Project Workflow](#️-project-workflow)
- [🚀 Installation & Setup](#-installation--setup)
- [📝 Usage](#-usage)
- [📊 Results & Visualization](#-results--visualization)
- [🔗 References](#-references)

## 🔍 Overview

Cataracts are a leading cause of blindness and visual impairment worldwide. Early and accurate detection is crucial for timely treatment. This project leverages transfer learning with the VGG19 model to classify retinal images as either **cataract** or **normal**.

The workflow includes data preprocessing, dataset balancing, model training, and evaluation, providing a comprehensive solution for automated cataract detection.

## 📚 Dataset

- **Source:** [Ocular Disease Recognition (ODIR-5K)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) from Kaggle.
- **Description:** The dataset contains retinal fundus images labeled for various ocular diseases, including cataract and normal cases.
- **Download:** You need a Kaggle account and API credentials to download the dataset.

## ⚙️ Project Workflow

1. **Environment Setup**
    - Installs required Python packages: `opendatasets`, `pandas`, etc.
    - Uses Google Colab with GPU acceleration for efficient training.

2. **Data Acquisition**
    - Downloads the ODIR-5K dataset from Kaggle using the `opendatasets` library.

3. **Data Preparation**
    - Reads and processes the provided CSV metadata.
    - Filters images labeled as "cataract" and "normal fundus".
    - Preprocesses images (resizing to 224x224 pixels).
    - Balances the dataset between cataract and normal images.

4. **Exploratory Data Analysis**
    - Visualizes class distribution.
    - Displays sample images from both classes.

5. **Model Preparation**
    - Splits data into train and test sets.
    - Uses transfer learning with the VGG19 architecture (from Keras/TensorFlow).
    - Prepares data for model ingestion (normalization, one-hot encoding).

6. **Training & Evaluation**
    - Trains the VGG19 model on the prepared dataset.
    - Evaluates performance (accuracy, loss, visualizations).

## 🚀 Installation & Setup

1. **Clone the Repository**
    ```bash
    git clone https://github.com/vishnupchopra/Cataract_Detection_VGG19.git
    cd Cataract_Detection_VGG19
    ```

2. **Install Dependencies**
    - The notebook uses the following Python packages:
        - opendatasets
        - pandas
        - numpy
        - matplotlib
        - seaborn
        - tensorflow
        - keras
        - scikit-learn
        - cv2
    - Install them using pip:
      ```bash
      pip install -r requirements.txt
      ```
      Or install them directly in your notebook using:
      ```python
      !pip install opendatasets pandas numpy matplotlib seaborn tensorflow keras scikit-learn opencv-python
      ```

3. **Dataset Download**
    - Make sure you have a Kaggle account.
    - Obtain your Kaggle API credentials and upload the `kaggle.json` file to your environment.
    - The notebook will prompt you to enter your Kaggle username and key if running in Colab.

## 📝 Usage

1. **Open the Notebook**

    - You can run the notebook directly in [Google Colab](https://colab.research.google.com/github/vishnupchopra/Cataract_Detection_VGG19/blob/main/Cataract_Detection_VGG19.ipynb) for free GPU access.

    - Or, run locally:
      ```bash
      jupyter notebook Cataract_Detection_VGG19.ipynb
      ```

2. **Run All Cells**

    - Follow the notebook cells sequentially—each section is well-commented for clarity.
    - The notebook will:
        - Download and preprocess the data
        - Visualize class distributions and samples
        - Train the VGG19 model
        - Output evaluation metrics and sample predictions

## 📊 Results & Visualization

- The notebook provides bar charts for class distribution and visual samples from both classes.
- Accuracy and loss metrics are displayed for both training and testing sets.
- Example predictions on images are visualized to demonstrate model performance.

## 🔗 References

- [ODIR-5K Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- [VGG19 Paper](https://arxiv.org/abs/1409.1556)
- [Keras VGG19 Documentation](https://keras.io/api/applications/vgg/#vgg19-function)

---

**Author:** [Vishnu Chopra](https://github.com/vishnupchopra)

Feel free to open issues or pull requests for suggestions and improvements! 🌟
