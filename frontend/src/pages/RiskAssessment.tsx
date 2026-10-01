import { useState } from 'react'
import { ArrowLeft, ArrowRight, Check, LoaderCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '../api'
import { Disclaimer, ErrorNotice, PageHeading } from '../components/Primitives'
import { useAppContext } from '../context/AppContext'
import type { PatientPayload } from '../types'

const steps = [
  { title: 'Personal information', hint: 'Core profile', fields: ['Age', 'Gender', 'Country'] },
  { title: 'Body & vital signs', hint: 'Measurements', fields: ['Height_cm', 'Weight_kg', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Heart_Rate'] },
  { title: 'Blood & metabolic health', hint: 'Lab values', fields: ['Blood_Glucose', 'HbA1c', 'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides'] },
  { title: 'Lifestyle', hint: 'Daily habits', fields: ['Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level', 'Sleep_Hours', 'Stress_Level', 'Diet_Quality', 'Exercise_Hours_Per_Week', 'Daily_Walking_Minutes'] },
  { title: 'Medical history & context', hint: 'Health history', fields: ['Family_History_Stroke', 'Family_History_Heart_Disease', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Previous_TIA', 'Atrial_Fibrillation', 'Chronic_Kidney_Disease', 'Medication_Adherence', 'Work_Type', 'Residence_Type', 'Air_Pollution_Exposure'] },
]

const labels: Record<string, string> = {
  Age: 'Age', Gender: 'Gender', Country: 'Country', Height_cm: 'Height (cm)', Weight_kg: 'Weight (kg)', BMI: 'BMI',
  Blood_Pressure_Systolic: 'Blood pressure · systolic', Blood_Pressure_Diastolic: 'Blood pressure · diastolic', Heart_Rate: 'Heart rate',
  Blood_Glucose: 'Blood glucose', HbA1c: 'HbA1c', Total_Cholesterol: 'Total cholesterol', HDL: 'HDL', LDL: 'LDL', Triglycerides: 'Triglycerides',
  Smoking_Status: 'Smoking status', Alcohol_Consumption: 'Alcohol consumption', Physical_Activity_Level: 'Physical activity level', Sleep_Hours: 'Sleep hours',
  Stress_Level: 'Stress level', Diet_Quality: 'Diet quality', Exercise_Hours_Per_Week: 'Exercise hours per week', Daily_Walking_Minutes: 'Daily walking minutes',
  Family_History_Stroke: 'Family history of stroke', Family_History_Heart_Disease: 'Family history of heart disease', Diabetes: 'Diabetes',
  Hypertension: 'Hypertension', Heart_Disease: 'Heart disease', Previous_TIA: 'Previous TIA', Atrial_Fibrillation: 'Atrial fibrillation',
  Chronic_Kidney_Disease: 'Chronic kidney disease', Medication_Adherence: 'Medication adherence', Work_Type: 'Work type', Residence_Type: 'Residence type',
  Air_Pollution_Exposure: 'Air pollution exposure',
}

const numericSteps = new Set(['Age', 'Height_cm', 'Weight_kg', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Heart_Rate', 'Blood_Glucose', 'HbA1c', 'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'Sleep_Hours', 'Exercise_Hours_Per_Week', 'Daily_Walking_Minutes'])
const binaryValues = new Set(['Family_History_Stroke', 'Family_History_Heart_Disease', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Previous_TIA', 'Atrial_Fibrillation', 'Chronic_Kidney_Disease'])
const allFeatures = steps.flatMap((step) => step.fields)

function emptyValues() {
  return Object.fromEntries(allFeatures.map((feature) => [feature, ''])) as Record<string, string>
}

export function RiskAssessmentPage() {
  const { modelInfo, serviceReady, saveAssessment } = useAppContext()
  const navigate = useNavigate()
  const [values, setValues] = useState<Record<string, string>>(emptyValues)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [stepIndex, setStepIndex] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [requestError, setRequestError] = useState<string | null>(null)

  if (!modelInfo) {
    return <><PageHeading eyebrow="Assessment" title="Patient information" description="Connect to the saved model to load its approved feature schema." /><div className="state-panel">Model schema is loading or the backend is unavailable.</div></>
  }

  const currentStep = steps[stepIndex]
  const optional = new Set(modelInfo.optional_features)

  const setField = (field: string, value: string) => {
    setValues((previous) => {
      const next = { ...previous, [field]: value }
      if (field === 'Height_cm' || field === 'Weight_kg') {
        const height = Number(next.Height_cm)
        const weight = Number(next.Weight_kg)
        next.BMI = height > 0 && weight > 0 ? (weight / ((height / 100) ** 2)).toFixed(1) : ''
      }
      return next
    })
    setErrors((previous) => ({ ...previous, [field]: '' }))
    setRequestError(null)
  }

  const validateFields = (fields: string[]) => {
    const found: Record<string, string> = {}
    for (const field of fields) {
      const raw = values[field]?.trim() ?? ''
      if (!raw) {
        if (!optional.has(field)) found[field] = 'This field is required.'
        continue
      }
      if (numericSteps.has(field)) {
        const parsed = Number(raw)
        const bounds = modelInfo.numeric_ranges[field]
        if (!Number.isFinite(parsed)) found[field] = 'Enter a valid number.'
        else if (bounds && (parsed < bounds.min || parsed > bounds.max)) {
          found[field] = `Use a value from ${bounds.min} to ${bounds.max} (observed model range).`
        }
      } else {
        const allowed = modelInfo.categorical_options[field] ?? []
        if (!allowed.includes(raw)) found[field] = 'Choose an available option.'
      }
    }
    const systolic = Number(values.Blood_Pressure_Systolic)
    const diastolic = Number(values.Blood_Pressure_Diastolic)
    if (fields.includes('Blood_Pressure_Systolic') && values.Blood_Pressure_Systolic && values.Blood_Pressure_Diastolic && systolic <= diastolic) {
      found.Blood_Pressure_Systolic = 'Systolic must be greater than diastolic in this dataset.'
    }
    setErrors((previous) => ({ ...previous, ...found }))
    return Object.keys(found).length === 0
  }

  const buildPayload = (): PatientPayload => Object.fromEntries(
    modelInfo.approved_features.map((feature) => {
      const raw = values[feature]?.trim() ?? ''
      if (!raw) return [feature, null]
      return [feature, numericSteps.has(feature) ? Number(raw) : raw]
    }),
  )

  const moveNext = () => {
    if (!validateFields(currentStep.fields)) return
    setStepIndex((index) => Math.min(index + 1, steps.length - 1))
  }

  const submitAssessment = async () => {
    const firstInvalidStep = steps.findIndex((step) => !validateFields(step.fields))
    if (firstInvalidStep >= 0) {
      setStepIndex(firstInvalidStep)
      return
    }
    if (!serviceReady) {
      setRequestError('The prediction service is not connected. Start the FastAPI backend and try again.')
      return
    }
    const payload = buildPayload()
    setSubmitting(true)
    setRequestError(null)
    try {
      const prediction = await api.predict(payload)
      saveAssessment(payload, prediction)
      navigate('/results')
    } catch (error) {
      setRequestError(error instanceof ApiError ? error.message : 'The prediction could not be completed.')
    } finally {
      setSubmitting(false)
    }
  }

  const renderField = (field: string) => {
    const label = labels[field] ?? field.replaceAll('_', ' ')
    const isOptional = optional.has(field)
    const choices = modelInfo.categorical_options[field]
    const range = modelInfo.numeric_ranges[field]
    const isAutoBmi = field === 'BMI' && Boolean(values.Height_cm && values.Weight_kg)
    const hasError = Boolean(errors[field])

    return (
      <div className={`field ${field === 'Country' ? 'field-full' : ''}`} key={field}>
        {binaryValues.has(field) && choices ? (
          <fieldset className="fieldset">
            <legend>{label}{isOptional && <span className="field-hint"> · optional</span>}</legend>
            <div className="choice-row mt-2" role="radiogroup" aria-label={label}>
              {choices.map((choice) => (
                <button key={choice} type="button" role="radio" aria-checked={values[field] === choice} onClick={() => setField(field, choice)} className={`choice-button ${values[field] === choice ? 'choice-button-selected' : ''}`}>
                  {values[field] === choice && <Check size={12} className="inline mr-1" />}{choice}
                </button>
              ))}
            </div>
          </fieldset>
        ) : (
          <>
            <div className="field-label-row">
              <label htmlFor={`field-${field}`}>{label}</label>
              {isOptional && <span className="field-hint">Optional</span>}
            </div>
            {choices ? (
              <select id={`field-${field}`} className="field-control" value={values[field]} onChange={(event) => setField(field, event.target.value)} aria-invalid={hasError}>
                <option value="">Select {label.toLowerCase()}</option>
                {choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
              </select>
            ) : (
              <input
                id={`field-${field}`}
                className="field-control"
                type="number"
                inputMode="decimal"
                step="any"
                min={range?.min}
                max={range?.max}
                value={values[field]}
                readOnly={isAutoBmi}
                onChange={(event) => setField(field, event.target.value)}
                placeholder={range ? `${range.min} – ${range.max}` : 'Enter value'}
                aria-invalid={hasError}
              />
            )}
          </>
        )}
        {errors[field] && <p className="field-error" role="alert">{errors[field]}</p>}
      </div>
    )
  }

  return (
    <>
      <PageHeading eyebrow="Risk assessment" title="Patient information" description="Complete the model-approved fields. Optional fields may be left blank and will use the saved training-time imputation." />
      {!serviceReady && <div className="mb-4"><ErrorNotice message="The saved model is not connected. You can complete the form, but assessment submission is unavailable until the backend is ready." /></div>}
      <div className="form-shell">
        <aside className="panel form-steps" aria-label="Assessment sections">
          <div className="form-steps-title">Assessment steps</div>
          {steps.map((step, index) => (
            <button key={step.title} className={`step-item ${index === stepIndex ? 'step-item-active' : ''} ${index < stepIndex ? 'step-item-complete' : ''}`} onClick={() => index <= stepIndex && setStepIndex(index)} disabled={index > stepIndex}>
              <span className="step-number">{index < stepIndex ? <Check size={13} /> : index + 1}</span>
              <span className="step-item-label">{step.title}<small>{step.hint}</small></span>
            </button>
          ))}
        </aside>

        <section className="panel form-main">
          <div className="form-progress"><span>Step {stepIndex + 1} of {steps.length}</span><div className="progress-track"><span style={{ width: `${((stepIndex + 1) / steps.length) * 100}%` }} /></div><span>{Math.round(((stepIndex + 1) / steps.length) * 100)}%</span></div>
          <p className="step-mobile-label">SECTION {stepIndex + 1} / {steps.length}</p>
          <div className="form-section-heading"><h2>{currentStep.title}</h2><p>{currentStep.hint} · values must fall within the model’s observed data range.</p></div>
          <div className="form-grid">{currentStep.fields.map(renderField)}</div>
          {stepIndex === steps.length - 1 && <div className="mt-5"><Disclaimer compact /></div>}
          {requestError && <div className="mt-4"><ErrorNotice message={requestError} /></div>}
          <div className={`form-actions ${stepIndex === 0 ? 'form-actions-end' : ''}`}>
            {stepIndex > 0 ? <button className="button button-secondary" onClick={() => setStepIndex((index) => Math.max(index - 1, 0))}><ArrowLeft size={15} /> Previous</button> : <span className="form-note">No identifier, target, or leakage fields are collected.</span>}
            {stepIndex < steps.length - 1 ? (
              <button className="button button-primary" onClick={moveNext}>Continue <ArrowRight size={15} /></button>
            ) : (
              <button className="button button-primary" onClick={submitAssessment} disabled={submitting || !serviceReady}>
                {submitting ? <><LoaderCircle size={15} className="animate-spin" /> Assessing</> : <>View assessment <ArrowRight size={15} /></>}
              </button>
            )}
          </div>
        </section>
      </div>
    </>
  )
}