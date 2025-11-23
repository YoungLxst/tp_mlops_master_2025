# Sentiment Polarity Classification – MLOps Project

This repository contains my work for the first sessions of an MLOps course. The objective is to build a complete machine-learning workflow for sentiment polarity prediction on textual reviews, including data preparation, modeling, evaluation, reproducibility, and experiment management.

---

## Project Overview

The goal of this project is to classify film reviews as **positive** or **negative**.
To achieve this, I built:

* A full project environment using **Git**, **Conda**, and **requirements management**
* Multiple **notebooks** for exploration, preprocessing, and model design
* A custom **PyTorch MLP classifier (PolarityNN)** used as a baseline model
* A scikit learn **LogistcRegression** model
* A preprocessing workflow based on **scikit-learn feature extraction**
* A modular structure to support future sessions (MLflow, hyperparameter optimization, pipelines…)

This project follows MLOps best practices: version control, environment isolation, documentation, and modular code.

---

## Repository Structure

```
mlops_project/
│── README.md
│── requirements.txt
│── notebooks/
│   ├── model_design.ipynb
│── data/
│   ├── train/
│   ├── test/
│   ├── validate/
│── model/
│   ├── Polaritynn.py
|   ├── trained/
│── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mlops_project.git
cd mlops_project
```

### 2. Create the Conda environment

```bash
conda create --name mlops python=3.11
conda activate mlops
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Make sure to update `requirements.txt` whenever a new library is added.

---

## Dataset

Download the dataset provided for the MLOps course:

➡️ [https://drive.google.com/file/d/1i2T30NH_PCwDJD2si7Rp_ZMPxPYBSatb/view](https://drive.google.com/file/d/1i2T30NH_PCwDJD2si7Rp_ZMPxPYBSatb/view)

Extract the archive and place the folders in:

```
data/train/
data/test/
data/validate/
```

Each folder contains text reviews and their associated polarity labels.

---

## Exploratory Data Analysis

EDA is performed in the notebook:

```
notebooks/exploratory_analysis.ipynb
```

The analysis includes:

* Inspection of missing values
* Distribution of polarity scores
* Basic statistics
* First inspection of text quality

Libraries used: **pandas**, **matplotlib**, **scikit-learn**.

---

## Data Preprocessing

Text preprocessing is based on **scikit-learn** feature extraction tools:

* `TfidfVectorizer`
* Removal of **French stop words**
* Vocabulary limitation (`max_features=5000`)
* Token normalization

Stop words are imported from **spaCy (fr_core_news_sm)**.

Documentation and implementation are also detailed inside the notebook:

```
notebooks/model_design.ipynb
```

---

## Model – PolarityNN (PyTorch MLP)

The main model implemented for this session is a custom PyTorch neural network named **PolarityNN**, located in:

```
src/model_polaritynn.py
```

### Model Architecture

* Input layer: 5000-dimensional bag-of-words vector
* Hidden layer 1: 128 units + ReLU + Dropout
* Hidden layer 2: 64 units + ReLU + Dropout
* Output layer: 1 neuron + Sigmoid (binary classification)

### Key features

* Custom training loop with progress monitoring
* Flexible optimizer & loss selection
* `.fit()`, `.score()`, `.predict()` implemented for sklearn-style usage
* Automatic model naming with timestamp
* Model saving utility

---

## Training the Model

Example training script (inside `model_design.ipynb`):

```python
from src.model_polaritynn import PolarityNN
from torch.utils.data import DataLoader, TensorDataset

model = PolarityNN(input_size=5000, hidden_size=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)

model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    learning_rate=0.001,
    optimizer="adam",
    criterion="bceloss"
)
```

---

## Evaluation

Metrics computed with `sklearn.metrics`:

* **Accuracy**
* **Precision**
* **Recall**

### Interpretation in this context

* **Accuracy**: fraction of correctly classified reviews
* **Precision**: among predicted “positive”, how many are truly positive
* **Recall**: among truly positive reviews, how many are detected

For sentiment classification, **precision and recall are more meaningful than accuracy**, especially in case of unbalanced classes.

---

## Next Steps

Upcoming features (next sessions):

* MLflow experiment tracking
* Full scikit-learn pipeline integration
* Model registry
* Training reproducibility
* Deployment and inference API (FastAPI)

---

Here is a clear and concise explanation in English, suitable for a TP/report:

---

# **Launching MLflow**

Once the setup is complete, starting the MLflow Tracking Server becomes very simple.
Instead of typing the full command manually each time, you can simply run one of the provided launcher scripts:

* **`start_mlflow.sh`** for Linux and macOS
* **`start_mlflow.bat`** for Windows
* **`start_mlflow.py`** for a fully cross-platform solution (works on Windows, Linux, and macOS)

Each script automatically:

1. Navigates to the correct directory
2. Ensures the required folders exist
3. Starts the MLflow server with the appropriate configuration

To launch MLflow, you only need to execute one of these scripts depending on your operating system. The server will then start and become accessible through the web interface at:

👉 **[http://localhost:5000](http://localhost:5000)**

This greatly simplifies the workflow and ensures that MLflow is always launched with consistent parameters.

# Author

**VANDENBERGHE ilian**  
MLOps Student – Université de Lille  
GitHub - @YoungLxst