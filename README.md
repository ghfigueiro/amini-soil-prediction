# 🌱 3rd Place Amini Soil Prediction Challenge – Technical Documentation  
https://zindi.africa/competitions/amini-soil-prediction-challenge

## 🧭 Objectives

The Amini Soil Prediction Challenge offers the opportunity to contribute to maximizing crop production in Africa while also developing models that could be adapted to other regions.  
To achieve this goal, I aimed to develop a model that generalizes soil nutrient predictions as effectively as possible.

---

## 🔄 ETL Process

- **Excluded features:** `latitude`, `longitude`, `pid`, and `site` were not used for model training, based on prior experience from the [AirQo African Air Quality Prediction Challenge](https://zindi.africa/competitions/airqo-african-air-quality-prediction-challenge).
- These features often lead to overfitting rather than generalization.

### ➕ New Features (suggested by GPT-4o by OpenAI):

- Fertility Index  
- SOC Bulk Density  
- Aridity Index

### 📂 External Datasets Used:

- `LANDSAT8_data_updated.csv`  
- `MODIS_MOD16A2_data.csv`  
- `MODIS_MOD11A1_data.csv`  
- `MODIS_MOD09GA_data.csv`  
- `MODIS_MOD13Q1_data.csv`

### 🧮 Feature Engineering:

- **LANDSAT8:** One feature per `QA_PIXEL` bit (0 to 15)  
- **MODIS + LANDSAT8:** Monthly averages (by `pid`, ignoring year)  
- **PCA:** Applied to all datasets (except train/test), imputing missing values with mean.

#### 📊 Number of PCA Components:

- `LANDSAT8`: 10  
- `MODIS_MOD16A2`: 13  
- `MODIS_MOD09GA`: 13  
- `MODIS_MOD13Q1`: 13  
- `MODIS_MOD11A1`: 10

Component count was selected based on public leaderboard performance.  
All PCA datasets were merged with `train` and `test` via `pid`.  
Missing values filled using **site-specific means**.

---

## 🤖 Data Modeling

### Random Forest:

- 80% training / 20% for tuning `ccp_alpha`.  
- Inside training: 5-fold cross-validation to reduce overfitting.

### LightGBM:

- 5-fold CV split **by site**.  
- In each fold, 80% sites for training, 20% for validation.  
- Early stopping based on RMSE.

**Outcome:**  
→ 5 models per nutrient and algorithm.

---

## 🧬 Ensembling and Postprocessing

```python
predictions_test_rf = mean(Random Forest predictions)
df_test_pred_lgbm = mean(LightGBM predictions)

final_prediction = (predictions_test_rf * 0.25 + df_test_pred_lgbm * 0.75) * 1.03
