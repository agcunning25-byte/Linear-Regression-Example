# Supervised ML Regression: Predicting Anime User Ratings

## Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for a regression task. The analysis is conducted on a dataset of Anime shows with the primary goal of predicting user ratings.

The project objectives are threefold:
1.  Build a regression model capable of predicting user ratings (on a 0-5 scale) with a reasonable degree of accuracy.
2.  Utilize the model's coefficients to identify and analyze the key features and variables that most significantly influence user ratings.
3.  Draw data-driven conclusions from these insights and outline a plan for future model enhancements.

This repository showcases skills in data cleaning, exploratory data analysis (EDA), feature engineering (log transformations, one-hot encoding, scaling), model selection (comparing multiple linear regression variants), and results interpretation.

## Dataset

The dataset used for this analysis originally contained 12,101 records and 44 features. After a thorough cleaning process to handle missing values (like `duration`) and remove redundant or irrelevant features (like `title` and `description`), the final dataset used for modeling consisted of **7,465 rows and 41 columns**.

* **Target Variable:** `Rating` (the average user rating on a 0-5 scale).
* **Key Features:** The dataset includes numerical and categorical features, such as:
    * `Media type` (e.g., TV, Movie, OVA)
    * `Primary studio`
    * `Duration` (in minutes)
    * `Watching` (number of users currently watching)
    * `WantWatch` (number of users who want to watch)
    * `Dropped` (number of users who dropped the series)
    * Genre/content tags (e.g., `tag_Comedy`, `tag_Action`, etc.)

## Machine Learning Pipeline & Skills

This notebook demonstrates a comprehensive workflow for tackling a regression problem, highlighting proficiency in the following areas:

### 1. Data Cleaning and Preprocessing
* **Handling Missing Values:** Assessed and dropped rows with missing data (e.g., in the `duration` column).
* **Duplicate Removal:** Identified and removed duplicate entries to ensure data quality.
* **Feature Dropping:** Removed text-based and high-cardinality features like `title` and `description` to simplify the model.

### 2. Exploratory Data Analysis (EDA)
* **Data Visualization:** Utilized `matplotlib` and `seaborn` to create custom functions for plotting distributions (`histogram_boxplot`) and frequencies (`labeled_barplot`).
* **Variable Analysis:** Analyzed numerical (continuous and binomial) and categorical variable distributions to understand the data's structure, identify skewness, and inform feature engineering steps.

### 3. Feature Engineering
* **Log Transformation:** Applied a `log1p` transformation to skewed numerical features (like `watching`, `wantWatch`, `dropped`) and the target variable (`Rating`). This step was critical for improving the linear relationship and model performance.
* **One-Hot Encoding:** Converted categorical features (`mediaType`, `studio_primary`) into numerical format using `pandas.get_dummies` so they could be used in the linear models.
* **Feature Scaling:** Standardized all numerical features using `StandardScaler` from scikit-learn to ensure all features were on a comparable scale, which is essential for regularized models.

### 4. Model Building & Evaluation
* **Model Selection:** Implemented and compared five different linear regression models to find the best fit:
    1.  Base Linear Regression
    2.  Linear Regression (with Log-Transformed variables)
    3.  Lasso Cross-Validation (`LassoCV`)
    4.  Ridge Cross-Validation (`RidgeCV`)
    5.  Elastic Net Cross-Validation (`ElasticNetCV`)
* **Train-Test Split:** Utilized `train_test_split` to create a robust validation strategy.
* **Hyperparameter Tuning:** Leveraged cross-validation models (`RidgeCV`, `LassoCV`, `ElasticNetCV`) to automatically find the best regularization strength (alpha).
* **Model Evaluation:** Assessed models using **R-squared (R²)** and **Root Mean Squared Error (RMSE)** metrics to compare their predictive power and error.

---

## Model Performance

The log transformation of variables provided a significant boost in performance, increasing the Test R² from 0.517 to ~0.645. The **Ridge Regression with Cross-Validation** was selected as the final model, as it offered the best R-squared and RMSE while providing robustness against potential multicollinearity.

| Model | Test R-squared | Test RMSE |
| :--- | :--- | :--- |
| Linear Regression (Original) | 0.5170 | 0.5886 |
| Linear Regression (Log) | 0.6445 | 0.5050 |
| Lasso CV (Log) | 0.6446 | 0.5050 |
| **Ridge CV (Log)** | **0.6447** | **0.5049** |
| Elastic Net CV (Log) | 0.6447 | 0.5049 |

---

## Key Findings & Feature Importance

By analyzing the coefficients from the final Ridge model, we identified the key drivers of Anime ratings:

1.  **Completion Behavior:** The strongest predictor by far was `Dropped` **(Coef: -0.51)**. This indicates a powerful negative relationship: the more users drop a show, the lower its rating.
2.  **Social Engagement:** `WantWatch` **(Coef: +0.34)** and `Watching` **(Coef: +0.32)** were the most significant positive predictors. This suggests that social hype and current popularity ("bandwagon" behavior) heavily influence ratings.
3.  **Production & Format:**
    * `Duration` **(Coef: +0.26)** had a strong positive impact, suggesting users rate longer-form content more highly.
    * `mediaType_Music Video` **(Coef: +0.23)** was also a strong positive factor.
    * Conversely, `mediaType_TV` **(Coef: -0.23)** and certain studios (e.g., `studio_primary_DLE` at **-0.29** and `studio_primary_OLM` at **-0.24**) were associated with lower ratings.

## Limitations and Future Enhancements

This analysis provides a strong baseline, but further improvements are possible:
* **Improve Performance:** An R² of 64% and an RMSE of ~0.5 indicates moderate performance. More advanced feature engineering (e.g., polynomial features, interaction terms) or consolidating redundant "tag" features could help.
* **Advanced Modeling:** Explore non-linear models (like Random Forest, Gradient Boosting, or Neural Networks) which may be able to capture more complex patterns in the data and improve predictive accuracy.

## Technologies Used

* **Python**
* **Pandas & NumPy** for data manipulation and numerical operations.
* **Scikit-learn** for feature preprocessing (`StandardScaler`, `train_test_split`) and modeling (`LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`).
* **Matplotlib & Seaborn** for data visualization and EDA.
* **Google Colab** (Jupyter Notebook environment).
