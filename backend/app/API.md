# FastAPI Usage

Run from the `backend` directory after installing `requirements.txt`:

```powershell
python -m uvicorn app.main:app --reload
```

Set `FRONTEND_ORIGINS` to a comma-separated list of approved frontend origins. The default allows the Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`.

The saved model pipeline is loaded once during application startup. Startup does not train or tune a model.

## Endpoints

- `GET /health`: service status and whether the saved pipeline loaded.
- `POST /predict`: predicted class and Low/Moderate/High probabilities.
- `GET /model-info`: model, target, feature policy, and recorded evaluation metrics.
- `POST /explain`: prediction plus local SHAP contribution data.
- `GET /explanations/global`: saved global SHAP importance and plot locations.
- `POST /explanations/prediction`: compatibility alias for the local SHAP response.

Pydantic request validation uses observed feature categories from the final feature configuration and numeric min/max values observed in the prepared dataset. Those bounds define the model's observed data support; they are not clinical limits. Missing values are accepted only for fields that were missing in the training data and are imputed by the saved preprocessing pipeline. Unknown fields (including `Patient_ID` and leakage columns) are rejected.

## Example Request

The request body is one JSON object containing the 35 approved features. Optional fields may be `null` when missing from the original training data. Categories are constrained to the saved configuration. The values below come from a real row in the training split. A complete request schema is published at `/docs`.

```json
{
  "Age": 88.0,
  "Gender": "Other",
  "Country": "Japan",
  "Height_cm": 164.4,
  "Weight_kg": null,
  "BMI": 43.0,
  "Blood_Pressure_Systolic": 146,
  "Blood_Pressure_Diastolic": 107,
  "Heart_Rate": 107,
  "Blood_Glucose": 132.1,
  "HbA1c": 9.8,
  "Total_Cholesterol": 149.2,
  "HDL": 32.7,
  "LDL": 202.4,
  "Triglycerides": 322.8,
  "Smoking_Status": "Former",
  "Alcohol_Consumption": "Never",
  "Physical_Activity_Level": "Low",
  "Sleep_Hours": 5.8,
  "Stress_Level": "Moderate",
  "Diet_Quality": "Poor",
  "Family_History_Stroke": "No",
  "Family_History_Heart_Disease": "Yes",
  "Diabetes": "No",
  "Hypertension": "No",
  "Heart_Disease": "No",
  "Previous_TIA": "No",
  "Atrial_Fibrillation": "No",
  "Chronic_Kidney_Disease": "No",
  "Medication_Adherence": "Excellent",
  "Exercise_Hours_Per_Week": 9.5,
  "Daily_Walking_Minutes": 41.0,
  "Work_Type": "Student",
  "Residence_Type": "Urban",
  "Air_Pollution_Exposure": "High"
}
```

## Example Prediction Response

```json
{
  "predicted_risk": "Moderate",
  "probability_low": 0.18,
  "probability_moderate": 0.72,
  "probability_high": 0.10,
  "probabilities": {
    "low": 0.18,
    "moderate": 0.72,
    "high": 0.10
  },
  "disclaimer": "Educational output only. This is not a medical diagnosis or a substitute for professional medical advice."
}
```

The response numbers above show the structure only. Call the live endpoint for model-generated probabilities. All predictions and SHAP results are educational and are not medical diagnoses.