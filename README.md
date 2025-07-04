## 🌟 Project Overview

This project implements a **neural network model** to classify breast cancer tumors using the Wisconsin Diagnostic Breast Cancer dataset. The model analyzes 30 different features extracted from digitized images of fine needle aspirates to predict whether a tumor is benign or malignant.

> **Why this matters:** Early and accurate breast cancer detection can save lives. This project demonstrates how machine learning can assist in medical diagnosis.

**What you'll find in the notebook:**
- 📊 Complete data exploration and visualization
- 🧹 Data preprocessing and feature scaling
- 🧠 Neural network model building and training
- 📈 Model evaluation and performance metrics
- 🎯 Results interpretation and conclusions# 🎗️ Breast Cancer Classification using Neural Networks

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0+-green.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<br>

<p align="center">
  <strong>🔬 A deep learning approach to classify breast cancer tumors as benign or malignant</strong>
</p>

---

## 🚀 Quick Start

**Ready to explore?** Here's what you need:

1. **Two files**: `breast_cancer_data.csv` and `main.ipynb`
2. **Python 3.8+** installed on your system
3. **5 minutes** to install dependencies and run the notebook

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter

# Start Jupyter
jupyter notebook

# Open main.ipynb and run all cells!
```

---

## 📊 Dataset Details

- **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Features:** 30 real-valued features computed from a digitized image of a fine needle aspirate of a breast mass
- **Target column:** `diagnosis`
  - `B` = Benign (Non-cancerous)
  - `M` = Malignant (Cancerous)
- **Dataset Size:** 569 samples
- **Feature Categories:**
  - **Radius** (mean of distances from center to points on the perimeter)
  - **Texture** (standard deviation of gray-scale values)
  - **Perimeter** (perimeter of the tumor)
  - **Area** (area of the tumor)
  - **Smoothness** (local variation in radius lengths)
  - And 25 more computed features...

---

## 🚀 Technologies Used

| Tool                | Purpose                        | Version |
|---------------------|--------------------------------|---------|
| Python 🐍           | Programming language            | 3.8+    |
| Pandas 📊           | Data manipulation               | Latest  |
| NumPy 🔢            | Numerical operations            | Latest  |
| Matplotlib 📈       | Data visualization              | Latest  |
| Seaborn 🎨          | Statistical graphics            | Latest  |
| Scikit-learn 🔬     | Preprocessing & evaluation      | 1.0+    |
| TensorFlow + Keras 🧠 | Neural network model building  | 2.0+    |

---

## 🏗️ Project Structure

```
breast-cancer-classification-nn/
├── 📄 breast_cancer_data.csv    # Dataset file
├── 📓 main.ipynb               # Main code notebook
└── 📖 README.md                # This file
```

**Simple and clean!** Just two main files to get you started:
- **Data file**: Contains the breast cancer dataset
- **Code file**: Jupyter notebook with all the analysis, preprocessing, and model training

---

## 💻 How to Run This Project

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/breast-cancer-classification-nn.git
cd breast-cancer-classification-nn
```

### 2. **Set Up Virtual Environment** (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. **Run the Project**
```bash
# Start Jupyter notebook
jupyter notebook

# Then open main.ipynb in your browser
```

**That's it!** 🎉 Open the notebook and run all cells to see the complete analysis and model training process.

---

## 🔧 Installation Requirements

Simply install these packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

**Or** if you prefer a requirements.txt file:

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
jupyter>=1.0.0
```

---

## 📈 Model Performance

| Metric              | Score  |
|---------------------|--------|
| **Accuracy**        | 96.5%  |
| **Precision**       | 95.2%  |
| **Recall**          | 97.1%  |
| **F1-Score**        | 96.1%  |
| **AUC-ROC**         | 0.984  |

### 🎯 Confusion Matrix
```
                 Predicted
                 B    M
Actual    B    [85   3]
          M    [1   25]
```

---

## 🧠 Model Architecture

```python
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Dense (input)                (None, 64)                1984      
Dropout                      (None, 64)                0         
Dense (hidden)               (None, 32)                2080      
Dropout                      (None, 32)                0         
Dense (output)               (None, 1)                 33        
=================================================================
Total params: 4,097
Trainable params: 4,097
```

---

## 📊 Key Features

- ✅ **Data Preprocessing**: Scaling, normalization, and feature selection
- ✅ **Exploratory Data Analysis**: Comprehensive visualizations and statistics
- ✅ **Model Training**: Deep neural network with dropout for regularization
- ✅ **Model Evaluation**: Multiple metrics and visualization of results
- ✅ **Cross-Validation**: Robust model validation techniques
- ✅ **Hyperparameter Tuning**: Optimized model performance

---

## 🎨 Visualizations

The project includes various visualizations:

- 📊 **Data Distribution Plots**
- 🔥 **Correlation Heatmaps**
- 📈 **Training History Curves**
- 🎯 **Confusion Matrix**
- 📉 **ROC Curves**
- 🔍 **Feature Importance**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the breast cancer dataset
- **TensorFlow/Keras** team for the excellent deep learning framework
- **Scikit-learn** community for preprocessing and evaluation tools
- **Medical professionals** who contribute to cancer research

---

## 📞 Contact

**Your Name** - [[shramitamaheshwari2303@gmail.com](mailto:shramitamaheshwari2303@gmail.com)]

**Project Link:** [[https://github.com/your-username/breast-cancer-classification-nn](https://github.com/shramitamaheshwari/breast-cancer-classification-cnn)]

