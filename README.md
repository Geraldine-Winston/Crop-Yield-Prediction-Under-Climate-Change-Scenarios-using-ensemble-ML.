# Crop Yield Prediction Under Climate Change Scenarios Using Ensemble Machine Learning

This project uses ensemble machine learning techniques to predict crop yield based on climate change scenarios. The models integrate climatic variables such as temperature, rainfall, CO₂ levels, and soil properties to forecast agricultural output.

## Project Structure

- `crop_yield_climate_data.csv`: Input dataset containing historical climate and yield data.
- `crop_yield_ensemble_model.pkl`: Trained ensemble model saved for future predictions.
- `crop_yield_climate_data.xlsx`: Sample dataset exported to Excel.

## Models Used

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Stacking Regressor (Meta-model)

## Installation

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

## How to Run

1. **Load the dataset**:
   ```python
   data = pd.read_csv('crop_yield_climate_data.csv')
   ```

2. **Preprocess the data**:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Train the ensemble model**:
   ```python
   stacked_model.fit(X_train, y_train)
   ```

4. **Make predictions and evaluate**:
   ```python
   y_pred = stacked_model.predict(X_test)
   print(mean_squared_error(y_test, y_pred))
   print(r2_score(y_test, y_pred))
   ```

5. **Save the model**:
   ```python
   import joblib
   joblib.dump(stacked_model, 'crop_yield_ensemble_model.pkl')
   ```

## Dataset Columns

- `temperature` (°C)
- `rainfall` (mm)
- `co2` (ppm)
- `soil_moisture` (%)
- `crop_yield` (tons/hectare)

## Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score
- Cross-validated R² Score

## Future Work

- Hyperparameter tuning with GridSearchCV
- Deployment as a web application using Streamlit
- Integration with climate scenario simulation data

## Author

Ayebawanaemi Geraldine Winston

---

**Note**: This is a sample project. Replace the sample data with your actual climate and crop yield datasets for real-world applications.
