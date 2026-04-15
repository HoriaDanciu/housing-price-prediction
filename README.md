# 🏠 House Price Prediction – Advanced Regression Techniques

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-House%20Prices-blue)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Final Kaggle Rank (as of April 2026):** 449 / 4524 (top 10%) 🎯

## 📌 Project Overview

This project solves the classic **House Prices: Advanced Regression Techniques** competition on Kaggle. The goal is to predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa, given 79 explanatory features describing various aspects of the property (lot size, quality of materials, basement area, garage, etc.).

The evaluation metric is **Root Mean Squared Error (RMSE)** between the logarithm of the predicted price and the logarithm of the actual sale price. To achieve a competitive score, I built an **ensemble of two gradient boosting models**: CatBoost and XGBoost, optimised with Optuna.

## 🧠 Key Achievements

- **Rank:** 449th out of 4,524 participants (top 10%)
- **RMSE (log scale):** ≈ 0.122
- **Ensemble blend:** 80% CatBoost + 20% XGBoost
- **Successfully handled:** 80+ columns, missing values, outliers, categorical encoding, and hyperparameter tuning.

## 🧹 Data Cleaning & Preprocessing

### Handling Missing Values (80+ columns)
- **Features where `NaN` means “absence”** (e.g., `Alley`, `PoolQC`, `Fence`, `FireplaceQu`) → filled with `'No [feature item]'`. This allowed models to learn that “no alley” correlates with lower price.
- **Basement & garage features** → `NaN` replaced with `'No Basement'` / `'No Garage'`.
- **Numerical columns** (`LotFrontage`, `MasVnrArea`, `GarageYrBlt`) → filled with median or 0.
- **Electrical** → filled with the mode (only 1 missing value).

### Feature Engineering
- **Log transformation** of `SalePrice` to reduce skewness (original distribution was heavily right‑skewed).
- **Ordinal encoding** for quality‑related columns (e.g., `ExterQual`, `KitchenQual`) using custom mappings (`Po`=1, `Fa`=2, `TA`=3, `Gd`=4, `Ex`=5).
- **One‑hot encoding** for remaining categorical variables (`Neighborhood`, `MSZoning`, `SaleType`, etc.) – 20+ columns.

## 🤖 Modelling

Two powerful gradient boosting models were trained:

| Model | Optimised Hyperparameters (via Optuna)                                                                                                 |
|-------|----------------------------------------------------------------------------------------------------------------------------------------|
| **CatBoost** | `iterations` = 500–6000, `learning_rate` = 0.005–0.08, `depth` = 3–8                                                                   |
| **XGBoost**  | `n_estimators` = 500–6000, `learning_rate` = 0.005–0.08, `colsample_bytree` = 0.2–0.6, `subsample` = 0.4–0.8, `min_child_weight` = 2–5 |

- **Cross‑validation:** 10‑fold stratified K‑Fold to evaluate RMSE on log‑transformed prices.
- **Hyperparameter tuning:** Optuna with median pruning (15-20 trials per model) to find the best parameters efficiently.

## 🏆 Ensemble & Final Predictions

The final predictions are a **weighted average** of the two models’ outputs (converted back from log scale)

final_pred = 0.80 * exp(pred_CatBoost) + 0.20 * exp(pred_XGBoost)

The 80/20 split was chosen based on validation performance (CatBoost consistently gave slightly lower RMSE).

## 📈 Results

- **CatBoost CV RMSE (log):** ~0.122
- **XGBoost CV RMSE (log):** ~0.125
- **Ensemble RMSE (log):** ~0.121

## 📚 What I Learned

### Technical Skills
1. **Handling real‑world messy data:** Missing values are not always errors – they often encode meaningful information (e.g., `NaN` in `Alley` means no alley). This dataset taught me to read data descriptions carefully.
2. **Ordinal vs. One‑Hot Encoding:** Not all categorical features are the same. Quality ratings (`ExterQual`) are ordinal, while neighbourhood names are nominal. Using the wrong encoding hurts performance.
3. **Log transformation for skewed targets:** Applying `np.log(SalePrice)` reduced skewness from ~12 to near 0, which improved model convergence and RMSE.
4. **Outlier detection:** Visualisation (`plt.scatter`) combined with domain knowledge is more effective than blind statistical rules (e.g., Z‑score). Removing extreme unrealistic points prevented overfitting.
5. **Hyperparameter tuning with Optuna:** Learned to set up search spaces, use pruning to save time, and interpret `best_params`. Optuna’s TPE algorithm is far more efficient than grid search.
6. **Ensembling:** Combining two different gradient boosting models (CatBoost and XGBoost) smoothed out individual errors and gave a small but meaningful improvement.

### Soft Skills & Process
- **Iterative workflow:** Start simple, get a baseline, then incrementally add complexity (cleaning → encoding → tuning → ensembling). Trying to do everything at once leads to confusion.
- **Debugging pipelines:** Encountered many errors (string columns, `inf` values, mismatched test columns) and learned to systematically check data types, missing values, and column alignment.
- **Efficiency:** Realised that 10–20 Optuna trials with 5‑fold CV often gives 90% of the benefit of 100 trials, saving hours of computation.
- **Documentation:** Keeping a clean notebook with markdown cells and a detailed README makes the project portfolio‑ready for recruiters.

## 🚀 How to Reproduce

1. Run all cells in `house_price_prediction.ipynb`.
2. The notebook will:
    - Load and preprocess the data
    - Perform EDA and outlier removal
    - Optimise CatBoost & XGBoost with Optuna
    - Train final models on the full training set
    - Generate `submission.csv` with ensemble predictions



## 📦 Dependencies

- Python 3.10+
- pandas, numpy
- matplotlib
- seaborn
- scikit-learn
- catboost
- xgboost
- optuna
- scipy



## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- The Ames Housing dataset was originally compiled by Dean De Cock.
- Competition hosted by Kaggle.

---

**Author:** Horia Danciu  
**Date:** April 2026  
**Contact:** horia_danciu_8@berkeley.edu


