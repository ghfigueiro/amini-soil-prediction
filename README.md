# 🌱 3rd Place – Amini Soil Prediction Challenge 2025 | Technical Documentation  
https://zindi.africa/competitions/amini-soil-prediction-challenge

## 🧭 Objectives

The Amini Soil Prediction Challenge offers the opportunity to contribute to maximizing crop production in Africa while also developing models that could be adapted to other regions.  
To achieve this goal, I focused on developing a model that generalizes soil nutrient predictions as effectively and robustly as possible.

---

## 🔄 ETL Process

- **Excluded features:** `latitude`, `longitude`, `pid`, and `site` were not used for model training, based on prior experience from the [AirQo African Air Quality Prediction Challenge](https://zindi.africa/competitions/airqo-african-air-quality-prediction-challenge).
  - These features often introduce data leakage or lead to overfitting, reducing the model's generalization capability.

### ➕ New Features (suggested by GPT-4o by OpenAI):

- **Fertility Index** – Composite index aggregating normalized CEC, pH, and SOC:
```python
# Normalize key soil indicators using z-score normalization
df['cec20_norm'] = (df['cec20'] - df['cec20'].mean()) / df['cec20'].std()
df['ph20_norm'] = (df['ph20'] - df['ph20'].mean()) / df['ph20'].std()
df['soc20_norm'] = (df['soc20'] - df['soc20'].mean()) / df['soc20'].std()

# Combine into a single fertility index
df['fertility_index'] = df['cec20_norm'] + df['ph20_norm'] + df['soc20_norm']
```

- **SOC Bulk Density** – Proxy for carbon storage efficiency in soil:
```python
df['SOC_BulkDensity'] = df['soc20'] / df['BulkDensity']
```

- **Aridity Index** – Climate dryness indicator:
```python
df['aridity_index'] = df['bio12'] / df['bio1']
```

---

### 📂 External Datasets Used

- `LANDSAT8_data_updated.csv`  
- `MODIS_MOD16A2_data.csv`  
- `MODIS_MOD11A1_data.csv`  
- `MODIS_MOD09GA_data.csv`  
- `MODIS_MOD13Q1_data.csv`

---

### 🧮 Feature Engineering

- **LANDSAT8:** One feature per `QA_PIXEL` bit (0 to 15)  
- **MODIS + LANDSAT8:** Monthly averages by `pid` (year ignored)  
- **PCA:** Applied to all external datasets after imputing missing values with the column mean  
  - `train` and `test` were **not** included in PCA computation  

#### 📊 Number of PCA Components

| Dataset              | # Components |
|----------------------|--------------|
| LANDSAT8             | 10           |
| MODIS_MOD16A2        | 13           |
| MODIS_MOD09GA        | 13           |
| MODIS_MOD13Q1        | 13           |
| MODIS_MOD11A1        | 10           |

Component counts were selected based on public leaderboard performance.  
All PCA datasets were merged with `train` and `test` via `pid`.  
Missing values in the final merged dataset were filled using **site-specific means**.

---

## 🤖 Data Modeling

### Random Forest

- 80% of the training data was used to train the model; 20% was reserved to tune `ccp_alpha`.
- Within the training split: 5-fold cross-validation to reduce overfitting.
- One model per nutrient target.

### LightGBM

- 5-fold cross-validation split **by site**  
  - In each fold: 80% of sites for training, 20% for validation
  - Early stopping based on RMSE
- One model per nutrient target.

---

## 🧬 Ensembling and Postprocessing

```python
# Final ensemble prediction
final_prediction = (
    0.25 * predictions_test_rf +
    0.75 * df_test_pred_lgbm
) * 1.03  # Scaling factor to improve leaderboard RMSE
```

- The final prediction is a weighted ensemble:
  - **25%** from Random Forest predictions
  - **75%** from LightGBM predictions
- A scaling factor of **1.03** was applied to adjust for slight underestimation observed during cross-validation.

---

## ✅ Final Notes

This solution balances performance and generalization by combining tree-based models, domain-specific feature engineering, and external remote sensing data.  
The methodology is flexible and can be extended to other geographies facing similar challenges in soil nutrient estimation.
