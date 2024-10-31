# Tuberculosis Detection Using Keras and PyTorch with LIME and Grad-CAM - README
## 1. Goal
The goal of this project is to build a machine learning model using Keras for predicting the presence of tuberculosis (TB) from chest X-ray images. Additionally, the project provides model explainability using LIME (Local Interpretable Model-agnostic Explanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the X-ray images influenced the model’s predictions. This combination ensures that the model not only makes accurate predictions but also provides interpretable explanations for its decisions.

## Installation
Clone the repository using git:
```bash
git clone https://github.com/Prajwal2212/Tuberculosis-Detection-Model.git
cd Tuberculosis-Detection-Model
```

## 2. Process
### 2.1. Tools and Libraries
#### Python: Programming language used for model development.
#### Keras (TensorFlow): Framework used to build and train the convolutional neural network (CNN) for tuberculosis prediction.
#### PyTorch: Used for implementing Grad-CAM visualization to highlight important regions in the X-ray images.
#### LIME (Local Interpretable Model-agnostic Explanations): Library for generating interpretable explanations for model predictions.
#### NumPy, Pandas: For data handling and preprocessing.
#### Matplotlib, Seaborn: For data visualization.
#### OpenCV: For image processing.
### 2.2. Data Collection and Preprocessing
#### Dataset:

The dataset consists of chest X-ray images, labeled as either TB-positive or TB-negative.\
The images are preprocessed by resizing, normalizing pixel values, and applying data augmentation (e.g., rotation, zoom) to improve model generalization.\
#### Data Splitting:

The dataset is split into training, validation, and testing sets to evaluate model performance and prevent overfitting.
### 2.3. Model Development
#### Building the CNN with Keras:

A convolutional neural network (CNN) is built using Keras, consisting of multiple convolutional layers, pooling layers, and fully connected layers.\
The model architecture is designed to extract features from X-ray images and classify them as either TB-positive or TB-negative.\
The output layer uses a sigmoid activation function for binary classification.
#### Training the Model:

The model is trained using the binary cross-entropy loss function and the Adam optimizer.\
Early stopping and learning rate scheduling are applied to enhance model performance and avoid overfitting.\
The training and validation accuracy and loss are monitored to fine-tune the model.
### 2.4. Model Explainability with LIME and Grad-CAM
#### LIME Explanation:

LIME is used to generate interpretable explanations by perturbing the input images and analyzing how the model’s predictions change.\
The output highlights the regions in the X-ray that most contribute to the prediction, helping clinicians understand the rationale behind the model's decision.
#### Grad-CAM Visualization:

Grad-CAM is implemented using PyTorch to produce heatmaps that highlight the areas of the X-ray image that the model focused on when making predictions.\
The heatmaps are overlaid on the original images, making it clear which regions contributed to the prediction.
### 2.5. Code Structure
#### Data Preparation: Code for loading and preprocessing the chest X-ray dataset.
#### Model Training (Keras): Includes the CNN architecture, training loop, and model evaluation.
#### LIME Explanation: Code to apply LIME and generate explanations for individual predictions.
#### Grad-CAM Visualization (PyTorch): Implementation of Grad-CAM to produce and display heatmaps over the input images.
## 3. Result
The developed model successfully predicts whether a chest X-ray indicates tuberculosis, with high accuracy and the added benefit of visual explanations for each prediction.

### Model Performance:
The model achieves strong performance metrics, including high accuracy, precision, and recall on the test set.\
Evaluation metrics (confusion matrix, ROC curve, F1-score) indicate that the model is effective at distinguishing between TB-positive and TB-negative cases.
### Explainability Insights:
LIME explanations provide granular, interpretable feedback on specific image regions that influenced the model’s decision, making it easier for medical practitioners to trust the predictions.\
Grad-CAM visualizations produce clear heatmaps over the X-ray images, highlighting the lung regions that are most relevant for identifying TB. These heatmaps align with medical knowledge of TB-affected areas, further validating the model’s predictions.
### Conclusion:
This project demonstrates the power of deep learning combined with explainability techniques for medical diagnosis. The model is both accurate and interpretable, making it a useful tool in assisting radiologists in the detection of tuberculosis. Future work could involve testing the model on a broader dataset and integrating additional explainability methods to enhance the model’s transparency and reliability.

### Future Improvements:
Extend the dataset to include diverse patient demographics and X-ray variations.\
Experiment with other CNN architectures (e.g., ResNet, EfficientNet) for better performance.\
Explore more advanced explainability techniques like SHAP (SHapley Additive exPlanations) for richer insights.\
This combination of prediction accuracy and interpretability makes the solution not only powerful but also practical for real-world medical applications.
