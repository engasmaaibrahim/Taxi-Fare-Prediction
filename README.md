# Taxi Fare Prediction System

## Overview

This project is a **machine learning web application** that predicts taxi fare prices based on trip details such as distance, time, and traffic conditions.

The system is built using **Django** and powered by a **LightGBM model** optimized with advanced techniques to achieve high prediction accuracy.

---

## Problem Statement

Estimating taxi fares manually can be inaccurate due to multiple influencing factors.

This project aims to:

* Predict taxi fares accurately using machine learning
* Provide real-time fare estimation for users
* Enhance decision-making for transportation services

---

## Features

* Predict taxi fare based on user input
* Real-time prediction results
* High accuracy using optimized LightGBM model
* Interactive and responsive user interface
* User rating system for feedback

---

## Model & Techniques

### Machine Learning:

* **LightGBM (Gradient Boosting)**
* Hyperparameter tuning using **Optuna**

### Data Preprocessing:

* **StandardScaler** → feature scaling
* **PCA** → dimensionality reduction

---

## Input Features

* Trip distance
* Time & duration
* Traffic conditions
* Additional trip-related attributes

---

## Technologies Used

* **Backend:** Django, Python
* **Frontend:** HTML5, CSS3, JavaScript
* **Machine Learning:** LightGBM, Scikit-learn, Optuna
* **Model Storage:** joblib

---

##  How to Run

### 1- Clone the repository

```bash id="r7r7v2"
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2️- Create and activate virtual environment

#### On Windows:

```bash id="y22qv3"
python -m venv env
env\Scripts\activate
```

---

### 3️- Install dependencies

```bash id="5wbjcw"
pip install -r requirements.txt
```

---

### 4️- Run the server

```bash id="a9ks9f"
python manage.py runserver
```


---


## Key Concepts

* Regression Models
* Gradient Boosting (LightGBM)
* Hyperparameter Optimization (Optuna)
* Feature Engineering
* Web Deployment with Django

---



## Author

**Asmaa Ibrahim**
AI & Machine Learning Engineer
