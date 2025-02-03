# README: Deep Learning for Medical Image Analysis

## Overview
This repository contains deep learning models implemented using TensorFlow and Keras for medical image analysis and classification. The models are trained on datasets imported from Kaggle, covering:
- **Brain Tumor Classification**
- **Breast Cancer Classification**
- **Lung Cancer Prediction**
- **Medical Image Segmentation using U-Net**

The notebook employs Monte Carlo training, U-Net for image segmentation, and Convolutional Neural Networks (CNNs) for classification tasks.

---

## Datasets

The datasets used in this project were imported from Kaggle using `opendatasets`. The following datasets are included:

1. **Brain Tumor MRI Dataset**
   - **Source:** Kaggle
   - **Path:** `/content/brain-tumor-mri-dataset/`
   - **Classes:** `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`
   - **Usage:** Used to train a CNN model for multi-class classification.

2. **Breast Cancer Dataset**
   - **Source:** Kaggle
   - **Path:** `/content/breast-cancer-dataset/breast-cancer.csv`
   - **Features:** Various patient attributes relevant to breast cancer.
   - **Usage:** Preprocessed and used as an input to a CNN model for breast cancer classification.

3. **Lung Cancer Dataset**
   - **Source:** Kaggle ("Cancer Patients and Air Pollution: A New Link")
   - **Usage:** Tabular dataset preprocessed and analyzed for lung cancer risk prediction.

4. **Dynamic Contrast-Enhanced MRI (DCE-MRI) Dataset** (Placeholder)
   - **Usage:** Used for training the U-Net model for medical image segmentation (synthetic dataset in current form).

---

## Code Explanation

### 1. **U-Net Model for Image Segmentation**
- The U-Net model is implemented for image segmentation tasks.
- Uses convolutional layers, batch normalization, and upsampling layers.
- Loss function: **Dice Similarity Coefficient (DSC)**.
- Performs **Monte Carlo Training** with multiple training runs for robust evaluation.

**Code Implementation:**
- `unet_model()`: Builds the U-Net model.
- `dice_coefficient()`: Computes DSC loss.
- `compile_model()`: Compiles the U-Net with Adam optimizer.
- `monte_carlo_training()`: Runs multiple training instances to analyze performance.
- `load_data()`: Loads MRI images (currently placeholder synthetic data).

### 2. **Brain Tumor Classification Using CNN**
- Images are preprocessed (grayscale conversion, resizing, and normalization).
- A CNN model is trained on the dataset to classify tumor types.
- Uses **Softmax activation** for multi-class classification.

**Code Implementation:**
- `load_images_from_directory()`: Loads and preprocesses MRI images.
- `build_model()`: Constructs a CNN model with convolutional layers and dropout.
- `model.fit()`: Trains the CNN model.
- `predict_sample()`: Predicts and visualizes test samples.

### 3. **Breast Cancer Classification Using CNN**
- Tabular data is reshaped into a 2D format to feed into a CNN.
- Uses **Label Encoding** for categorical variables.
- Standardizes features using **StandardScaler**.
- Trained using **Sparse Categorical Crossentropy Loss**.

**Code Implementation:**
- `build_model()`: Constructs the CNN model.
- `train_test_split()`: Splits the dataset into train, validation, and test sets.
- `model.fit()`: Trains the CNN model.
- `predict_sample()`: Tests predictions on sample cases.

### 4. **Lung Cancer Dataset Analysis**
- Loads Kaggle dataset on cancer patients and air pollution.
- Extracts key features and applies preprocessing.
- Placeholder for further analysis and model development.

**Code Implementation:**
- `od.download()`: Downloads dataset from Kaggle.
- `pd.read_csv()`: Loads data into a pandas DataFrame.

---

## How to Run the Notebook

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Kaggle Datasets**
   - Ensure `kaggle.json` API key is set up.
   - Run:
     ```python
     import opendatasets as od
     od.download("https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link")
     ```

4. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   - Open the `.ipynb` file and run cells sequentially.

---

## Results
- **Brain Tumor Classification:** Achieves high accuracy in identifying tumor types.
- **Breast Cancer Classification:** Successfully distinguishes between malignant and benign cases.
- **Lung Cancer Dataset:** Provides insights into cancer and environmental factors.
- **U-Net Image Segmentation:** Demonstrates effective segmentation using Monte Carlo evaluation.

---

## Future Work
- Improve dataset augmentation techniques for MRI classification.
- Implement more advanced segmentation models for medical imaging.
- Apply deep learning explainability techniques (Grad-CAM, SHAP) to interpret CNN decisions.
- Explore additional Kaggle medical datasets for better model generalization.

---

## License
This project is released under the MIT License.

---

## Acknowledgments
- Kaggle for providing datasets.
- TensorFlow and Keras for deep learning frameworks.
- OpenCV for image processing utilities.

---
