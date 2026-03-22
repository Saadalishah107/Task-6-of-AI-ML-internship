# Task 6: House Price Prediction
**DevelopersHub Corporation — AI/ML Engineering Internship**

## Objective
Predict house prices based on property and demographic features using regression models. Evaluate and compare model performance using MAE, RMSE, and R².

## Dataset
- **Name:** California Housing Dataset
- **Source:** `sklearn.datasets.fetch_california_housing`
- **Size:** 20,640 samples × 8 features
- **Target:** `MedHouseVal` — Median house value (in $100,000s)

## Features Used
| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

## Tools & Libraries
- Python 3.10
- pandas, numpy
- scikit-learn (LinearRegression, GradientBoostingRegressor)
- matplotlib, seaborn

## What's Inside the Notebook
| Step | Description |
|------|-------------|
| EDA | Target distribution, log transform, correlation heatmap, scatter plots |
| Geo Visualization | Geographic price map using Latitude/Longitude |
| Preprocessing | StandardScaler, 80/20 train/test split |
| Model 1 | Linear Regression (baseline) |
| Model 2 | Gradient Boosting Regressor (200 estimators, depth=4) |
| Evaluation | MAE, RMSE, R², Actual vs Predicted plots, Residual plots |
| Feature Importance | Ranked bar chart from Gradient Boosting |

## Model Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | ~0.53 | ~0.73 | ~0.60 |
| Gradient Boosting | ~0.37 | ~0.52 | ~0.81 |

## Key Results & Findings
- **Gradient Boosting significantly outperforms Linear Regression** (R² of 0.81 vs 0.60), confirming strong non-linear patterns in housing data
- **Median Income (`MedInc`)** is by far the most important predictor of house price
- **Latitude and Longitude** rank highly — coastal California areas (Bay Area, LA) command premium prices
- **Linear Regression underpredicts** very high-value homes, visible in residual plots
- **Gradient Boosting residuals** are more evenly distributed around zero, indicating a better fit

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Task6_House_Price_Prediction.ipynb
```
Run all cells from top to bottom. Dataset loads automatically via sklearn.
