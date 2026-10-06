# 🚴‍♂️ Food Delivery Time Forecast Project

An end-to-end Machine Learning and Data Science project that predicts total food delivery duration (in minutes) based on logistics, weather, traffic, and courier features.

Rather than relying purely on distance-based estimates, this project leverages regression models to account for real-world factors—such as rain, heavy traffic, driver experience, and kitchen prep time—to deliver accurate Estimated Time of Arrival (ETA) forecasts.

---

## 📌 Project Objectives

Accurate ETA predictions in online food delivery platforms directly improve customer satisfaction, optimize dispatch operations, and reduce delivery delays. Key goals include:

* Conducting **Exploratory Data Analysis (EDA)** to identify key drivers of delivery delay.
* Cleaning and preprocessing complex tabular data containing missing values and mixed types.
* Training and evaluating multiple regression models using metrics like $R^2$, $MAE$, $MSE$, and $RMSE$.
* Building a structured, reproducible machine learning workflow.

---

## 📊 Dataset Overview

The dataset (`Food_Delivery_Times.csv`) consists of **1,000 order records** with **9 features**:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Order_ID` | Numerical (`int64`) | Unique identifier for each order |
| `Distance_km` | Numerical (`float64`) | Delivery distance in kilometers |
| `Weather` | Categorical (`object`) | Weather conditions (*Clear, Rainy, Foggy, Snowy, Windy*) |
| `Traffic_Level` | Categorical (`object`) | Traffic intensity (*Low, Medium, High*) |
| `Time_of_Day` | Categorical (`object`) | Time frame of order (*Morning, Afternoon, Evening, Night*) |
| `Vehicle_Type` | Categorical (`object`) | Courier transport mode (*Scooter, Bike, Car*) |
| `Preparation_Time_min` | Numerical (`int64`) | Restaurant food preparation time in minutes |
| `Courier_Experience_yrs` | Numerical (`float64`) | Courier experience in years |
| **`Delivery_Time_min`** | **Numerical (`int64`)** | **Target Variable:** Total delivery duration in minutes |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning Pipeline:** `scikit-learn`, `xgboost`
  * *Preprocessing:* `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder`, `SimpleImputer`
  * *Algorithms:* `LinearRegression`, `RandomForestRegressor`, `XGBRegressor`
  * *Evaluation Metrics:* $MAE$, $MSE$, $RMSE$, $R^2$ Score

---

## 🔍 Key Insights from Exploratory Data Analysis (EDA)

1. **Primary Drivers:** Delivery distance (`Distance_km`) exhibits the strongest positive correlation with total delivery time ($r \approx 0.78$), followed by food prep time (`Preparation_Time_min`) ($r \approx 0.31$).
2. **Missing Values:** Approximately 3% of values are missing across columns (`Weather`, `Traffic_Level`, `Time_of_Day`, `Courier_Experience_yrs`). These are handled via median/mode imputation strategies.
3. **Target Distribution:** Average delivery duration is roughly **56.7 minutes** with a standard deviation of **22 minutes**.

---

## ⚙️ Machine Learning Workflow

1. **Data Inspection & Cleaning:**
   * Checking data types, null values, and summary statistics.
   * Treating outliers and filling missing values using `SimpleImputer`.
2. **Feature Engineering & Encoding:**
   * Ordinal encoding for ordered categories (`Traffic_Level`).
   * One-Hot encoding for nominal categories (`Weather`, `Time_of_Day`, `Vehicle_Type`).
   * Scaling numerical features using `StandardScaler`.
3. **Model Training:**
   * Splitting data into 80% Training and 20% Testing sets.
   * Benchmarking linear models against tree-based ensembles (Random Forest, XGBoost).
4. **Performance Evaluation:**
   * Evaluating models using test set metrics ($R^2$, $MAE$, $RMSE$).
