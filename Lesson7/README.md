
# Lesson 2: Linear Regression

## Predicting House Prices with Multiple Features

This lesson introduces **Linear Regression** using a housing dataset with both numeric and categorical features. Students learn how to build a complete machine learning workflow using **scikit-learn Pipelines** and **ColumnTransformers**, which will become the standard pattern used throughout the course.

## Learning Objectives

By the end of this lesson, students should be able to:

- Perform a train/test split for machine learning.
- Distinguish between numerical and categorical variables.
- Apply one-hot encoding to categorical features.
- Build a preprocessing pipeline using `ColumnTransformer`.
- Create an end-to-end machine learning workflow using `Pipeline`.
- Train a Linear Regression model.
- Evaluate model performance using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Interpret regression coefficients.
- Understand how categorical variables appear as multiple coefficients after one-hot encoding.

---

## Dataset

The dataset contains information about homes including:

| Feature | Description |
|----------|-------------|
| sqft | Square footage |
| bedrooms | Number of bedrooms |
| age | Age of home |
| neighborhood | Neighborhood category |
| price | Home selling price (target variable) |

---

## Workflow

### 1. Load Data

Import the housing dataset and inspect the records.

### 2. Train/Test Split

Separate data into training and testing sets before any preprocessing.

### 3. Preprocessing

Use a `ColumnTransformer` to:

- Pass numerical variables unchanged
- Apply `OneHotEncoder` to categorical variables

### 4. Build Pipeline

Create a reusable workflow:

```python
Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])
```

### 5. Train Model

Fit the model using the training data.

### 6. Evaluate Performance

Measure predictive accuracy on the test set using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

### 7. Interpret Results

Extract and examine coefficients to understand how each variable influences house price.

---

## Key Concepts

### Linear Regression

Linear Regression attempts to model the relationship:

\[
Price = b_0 + b_1X_1 + b_2X_2 + \cdots + b_nX_n
\]

where:

- \(b_0\) = intercept
- \(b_i\) = coefficient for feature \(i\)

### One-Hot Encoding

Categorical variables cannot be used directly in most machine learning models.

Example:

| Neighborhood |
|-------------|
| Urban |
| Suburban |
| Rural |

becomes:

| Urban | Suburban | Rural |
|--------|--------|--------|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

---

## Skills Practiced

- Data preprocessing
- Feature engineering
- Train/test methodology
- Pipeline construction
- Regression modeling
- Model evaluation
- Model interpretation

---

## Files

- `Lesson2_LinearRegression.ipynb` — Main notebook
- `homes.csv` — Housing dataset
- `lesson2_linear_regression.pdf` — Companion notes

---

## Next Lesson

In the next lesson, we will compare Linear Regression to more advanced machine learning models and continue using the same Pipeline + ColumnTransformer framework introduced here.

---

## Course Information

**Course:** INFO 648 – Introduction to Analytics  
**Instructor:** Ben Marlin, PhD, CAP  
**VCU School of Business**
