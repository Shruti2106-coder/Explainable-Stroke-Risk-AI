import { ArrowRight, RotateCcw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'
import { Disclaimer, EmptyState, PageHeading, Panel, RiskBadge } from '../components/Primitives'
import type { RiskClass } from '../types'

const riskColors: Record<RiskClass, string> = { Low: '#4CAF7D', Moderate: '#E5A84B', High: '#D96B6B' }
const summaryLabels: Record<string, string> = {
  Age: 'Age', Gender: 'Gender', Country: 'Country', BMI: 'BMI', Blood_Pressure_Systolic: 'Systolic pressure',
  Blood_Pressure_Diastolic: 'Diastolic pressure', Heart_Rate: 'Heart rate', Smoking_Status: 'Smoking', Diabetes: 'Diabetes',
  Hypertension: 'Hypertension', Previous_TIA: 'Previous TIA', Atrial_Fibrillation: 'Atrial fibrillation',
}

export function ResultsPage() {
  const { patient, prediction } = useAppContext()
  if (!patient || !prediction) {
    return <><PageHeading eyebrow="Results" title="Assessment results" description="Your saved result appears here after an assessment." /><EmptyState title="No assessment yet" description="Complete the patient information form to see the model output and class probabilities." /></>
  }

  const risk = prediction.predicted_risk
  const probabilityData = [
    { name: 'Low', probability: prediction.probability_low },
    { name: 'Moderate', probability: prediction.probability_moderate },
    { name: 'High', probability: prediction.probability_high },
  ]
  const highestProbability = Math.max(...probabilityData.map(({ probability }) => probability))
  const summary = Object.entries(summaryLabels).filter(([key]) => patient[key] !== null && patient[key] !== undefined && patient[key] !== '')

  return (
    <>
      <PageHeading eyebrow="Assessment result" title="Your model output" description="This is the finalized model’s estimate for the information submitted. Review the explanation before drawing conclusions." action={<Link to="/assessment" className="button button-secondary"><RotateCcw size={15} /> New assessment</Link>} />
      <div className="result-hero">
        <section className="result-overview">
          <p className="result-risk-label">Predicted stroke-risk class</p>
          <h2 className="result-risk-title">{risk.toUpperCase()}</h2>
          <div className="mt-3"><RiskBadge risk={risk} large /></div>
          <p>The class with the highest model probability for this input. This is an educational/decision-support assessment and is not a medical diagnosis.</p>
          <Link to="/explanation" className="button button-primary w-fit">Understand this prediction <ArrowRight size={15} /></Link>
        </section>
        <Panel className="probability-panel">
          <h2>Risk probability</h2>
          <p className="muted-copy mt-1">Model output across the three target classes.</p>
          <div className="probability-list" aria-label="Risk probabilities">
            {probabilityData.map(({ name, probability }) => {
              const isHighest = probability === highestProbability
              const percent = probability * 100
              return (
                <div className={`probability-row${isHighest ? ' probability-row-highest' : ''}`} key={name}>
                  <div className="probability-row-heading">
                    <span className="probability-class" style={{ color: riskColors[name as RiskClass] }}>{name}</span>
                    <strong className="probability-value">{percent.toFixed(1)}%</strong>
                    {isHighest && <span className="probability-highest-label">Highest probability</span>}
                  </div>
                  <div
                    className="probability-track"
                    role="progressbar"
                    aria-label={`${name} probability`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={percent}
                  >
                    <span className="probability-fill" style={{ width: `${percent}%`, backgroundColor: riskColors[name as RiskClass] }} />
                  </div>
                </div>
              )
            })}
          </div>
        </Panel>
      </div>

      <section className="home-section">
        <div className="section-title"><div><h2>Patient summary</h2><p>Values supplied for this model request.</p></div><Link to="/explanation" className="link-arrow">View feature contributions <ArrowRight size={14} /></Link></div>
        <div className="panel summary-grid">
          {summary.map(([key, label]) => <div className="summary-item" key={key}><span>{label}</span><strong>{String(patient[key])}</strong></div>)}
        </div>
      </section>
      <div className="result-disclaimer"><Disclaimer /></div>
    </>
  )
}