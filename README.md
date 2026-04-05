# USA House Price Prediction

## Project overview
This project aims to predict house prices from a US residential real estate dataset using machine learning.

The workflow covers:
- data cleaning
- exploratory data analysis (EDA)
- feature engineering
- preprocessing with sklearn pipelines
- model comparison
- cross-validation
- overfitting analysis
- final model selection

The final goal is not only to obtain good predictive performance, but also to identify the most robust model in terms of generalization.

## Problem statement
Can we reliably predict house prices from structural, location, and quality-related features, while identifying the model that offers the best balance between predictive performance, robustness, and generalization?

## Dataset
The dataset contains around 4,000 residential properties and includes:
- structural variables: bedrooms, bathrooms, floors
- surface variables: sqft_living, sqft_lot, sqft_above, sqft_basement
- quality variables: view, condition, waterfront
- temporal variables: date, yr_built, yr_renovated
- location variables: city, statezip

Target variable:
- price

## Project workflow

### 1. Data cleaning
The dataset was first audited to identify:
- missing values
- duplicated rows
- fake missing values
- inconsistent zeros

The following cleaning decisions were made:
- remove rows where price = 0
- remove rows where bedrooms = 0
- remove rows where bathrooms = 0
- drop country because it is constant
- drop street because of its very high cardinality

### 2. Exploratory Data Analysis
The EDA highlighted several important findings:
- price is strongly right-skewed
- sqft_living is the strongest numerical predictor
- location has a major impact on price
- waterfront and view clearly increase property value
- several variables are highly correlated, especially:
  - sqft_living and sqft_above
  - bedrooms, bathrooms and total_rooms
  - yr_built and home_age

### 3. Feature engineering
Several new features were created to enrich the signal:
- sale_year, sale_month, sale_day
- is_renovated
- has_basement
- home_age
- sqft_ratio
- total_rooms

A second feature engineering step added more advanced features:
- bath_bed_ratio
- living_x_view
- living_x_waterfront
- living_x_condition
- log_sqft_living
- log_sqft_lot
- home_age_sq
- quality_score

### 4. Preprocessing
The preprocessing was integrated directly into sklearn pipelines:
- numerical variables:
  - median imputation
  - scaling for linear models
- categorical variables:
  - most frequent imputation
  - one-hot encoding

Two preprocessors were used:
- one for linear models
- one for tree-based models

### 5. Models compared
The following models were tested:
- DummyRegressor
- Ridge
- RandomForestRegressor
- GradientBoostingRegressor
- XGBoost
- LightGBM

### 6. Evaluation strategy
The target was log-transformed using log1p(price) because of the strong right skew.

Models were evaluated using:
- test set metrics:
  - MAE
  - RMSE
  - R2
  - MAPE
- 5-fold cross-validation
- overfitting analysis using the train-validation gap

## Final results

### Test set
| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| Ridge_v2 | 97,640 | 166,258 | 0.675 | 20.29% |
| GradientBoosting_v2 | 100,973 | 168,952 | 0.665 | 20.75% |
| XGBoost | 100,304 | 177,988 | 0.628 | 21.61% |
| LightGBM | 106,113 | 187,164 | 0.589 | 21.76% |
| RandomForest_v2 | 122,839 | 194,041 | 0.558 | 25.38% |

### Cross-validation
| Model | CV R2 mean | CV R2 std | Train R2 mean | Gap train-CV |
|---|---:|---:|---:|---:|
| Ridge_v2 | 0.716 | 0.042 | 0.728 | 0.011 |
| GradientBoosting_v2 | 0.712 | 0.042 | 0.779 | 0.067 |
| XGBoost | 0.705 | 0.053 | 0.908 | 0.203 |
| LightGBM | 0.694 | 0.045 | 0.925 | 0.231 |
| RandomForest_v2 | 0.632 | 0.045 | 0.712 | 0.080 |

## Final model selection
The final selected model is **Ridge_v2**.

Why?
- best overall test performance
- best cross-validation mean score
- very small train-validation gap
- strongest robustness and generalization

This result is particularly interesting because it shows that a well-prepared linear model can outperform more complex boosting models when feature engineering and validation are done carefully.

## Key takeaways
- good feature engineering can matter as much as model complexity
- more complex models do not automatically perform better
- cross-validation is essential to detect overfitting
- Ridge_v2 offered the best balance between performance and robustness

## Repository structure
Recommended structure:

```text
house-price-prediction/
├── data/
│   └── USA Housing Dataset.csv
├── notebooks/
│   └── Pred_House_price.ipynb
├── images/
│   ├── eda_price_distribution.png
│   ├── model_comparison.png
│   ├── overfitting_gap.png
│   └── real_vs_predicted.png
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```

Then open:
- notebooks/Pred_House_price.ipynb



## Author
Walid Si khelifa
