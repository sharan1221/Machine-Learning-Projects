# House Price Prediction — Linear Regression

Built a Linear Regression model to predict house prices based 
on 16 features including size, quality, location, and more.

## Results
- R² Score : 0.81
- RMSE     : $37,865
- Dataset  : 1,460 houses (Kaggle)

## What I did
- Loaded and explored real-world housing data
- Selected 16 meaningful features from 80+ columns
- Preprocessed data (handled missing values, encoded categories)
- Split data 80/20 for training and testing
- Trained Linear Regression model using scikit-learn
- Evaluated using R² Score and RMSE
- Built interactive price predictor for new house inputs

## Tech Stack
Python · scikit-learn · pandas · NumPy · matplotlib

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python house_price_prediction.py
```
Download dataset from:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
