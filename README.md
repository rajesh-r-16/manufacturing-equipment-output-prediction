# Manufacturing Equipment Output Prediction

An end-to-end machine learning application for predicting the hourly production output of injection molding machines using machine operating parameters, material properties, environmental conditions, and equipment/operator factors.

## 📌 Project Overview

Manufacturing companies need to understand how machine operating conditions affect production output. This project develops a supervised machine learning regression system to predict the number of parts produced per hour (`Parts_Per_Hour`) from manufacturing equipment parameters.

The project includes data preprocessing, feature engineering, categorical encoding, feature scaling, regression model training, model evaluation, and deployment using Streamlit.

## 🎯 Objectives

- Predict hourly machine production output.
- Analyze the relationship between machine parameters and production efficiency.
- Apply data preprocessing and feature engineering techniques.
- Compare Linear Regression, Ridge Regression, and Lasso Regression.
- Evaluate models using standard regression metrics.
- Provide an interactive web application for real-time prediction.
- Support production planning and identification of underperforming machine conditions.

## 🏭 Problem Statement

The project focuses on injection molding machines used to produce plastic components.

The objective is to predict hourly production output based on parameters such as:

- Injection temperature
- Injection pressure
- Cycle time
- Cooling time
- Material viscosity
- Ambient temperature
- Machine age
- Operator experience
- Maintenance hours
- Shift
- Machine type
- Material grade
- Day of week

The predicted output can be used to understand machine performance and support production planning.

## 📊 Dataset

The project uses a synthetic manufacturing dataset based on real-world manufacturing principles.

### Target Variable

`Parts_Per_Hour`

### Dataset Features

| Feature | Description |
|---|---|
| Injection_Temperature | Injection temperature of the machine |
| Injection_Pressure | Injection pressure |
| Cycle_Time | Machine cycle time |
| Cooling_Time | Cooling duration |
| Material_Viscosity | Material viscosity |
| Ambient_Temperature | Environmental temperature |
| Machine_Age | Age of the machine |
| Operator_Experience | Operator experience |
| Maintenance_Hours | Maintenance-related hours |
| Shift | Working shift |
| Machine_Type | Type of machine |
| Material_Grade | Material grade |
| Day_of_Week | Day of operation |
| Temperature_Pressure_Ratio | Engineered dataset feature |
| Total_Cycle_Time | Engineered dataset feature |
| Efficiency_Score | Engineered dataset feature |
| Machine_Utilization | Engineered dataset feature |

Time-related information can also be extracted from the timestamp, such as hour, month, and day.

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Processing the timestamp information.
2. Handling missing values.
3. Encoding categorical variables.
4. Scaling numerical features.
5. Preparing the target variable.
6. Splitting the dataset into training and testing sets.

The categorical variables include:

- Shift
- Machine_Type
- Material_Grade
- Day_of_Week

The project specification uses an 80:20 train-test split.

## 🧠 Machine Learning Models

Three regression models are implemented:

### 1. Linear Regression

A baseline regression model used to establish a relationship between the input variables and hourly production output.

### 2. Ridge Regression

A regularized linear regression model that helps control the influence of correlated features.

### 3. Lasso Regression

A regularized regression model that can reduce the influence of less important features.

## 📈 Model Evaluation

The models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### Evaluation Metrics

**R² Score**

Measures how much of the variation in the target variable is explained by the model.

**MAE**

Measures the average absolute difference between actual and predicted values.

**MSE**

Measures the average squared prediction error.

**RMSE**

Represents the square root of MSE and expresses prediction error in the same unit as the target variable.

## 🖥️ Streamlit Application

The trained models are deployed through an interactive Streamlit application.

The application allows users to:

- Enter manufacturing parameters.
- Select a prediction model.
- Generate hourly production predictions.
- Compare predictions from multiple regression models.
- Upload manufacturing data for batch prediction.
- Download prediction results.

### Application Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Ensemble prediction

The ensemble option combines predictions from the available trained regression models.

## 🏗️ System Workflow

```text
Manufacturing Data
        ↓
Data Loading
        ↓
Data Preprocessing
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Regression Models
        ↓
Linear / Ridge / Lasso
        ↓
Model Evaluation
        ↓
Saved Trained Models
        ↓
Streamlit Application
        ↓
Parts Per Hour Prediction