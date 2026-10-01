import { ArrowRight, RotateCcw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
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
  const summary = Object.entries(summaryLabels).filter(([key]) => patient[key] !== null && patient[key] !== undefined && patient[key] !== '')

  return (
    <>
      <PageHeading eyebrow="Assessment result" title="Your model output" description="This is the finalized model’s estimate for the information submitted. Review the explanation before drawing conclusions." action={<Link to="/assessment" className="button button-secondary"><RotateCcw size={15} /> New assessment</Link>} />
      <div className="result-hero">
        <section className="result-overview">
          <p className="result-risk-label">Predicted stroke-risk class</p>
          <h2 className="result-risk-title">{risk.toUpperCase()}</h2>
          <div className="mt-3"><RiskBadge risk={risk} large /></div>
          <p>The class with the highest model probability for this input. This output is educational and is not a diagnosis.</p>
          <Link to="/explanation" className="button button-primary w-fit">Understand this prediction <ArrowRight size={15} /></Link>
        </section>
        <Panel className="probability-panel">
          <h2>Class probabilities</h2>
          <p className="muted-copy mt-1">Model output across the three target classes.</p>
          <div className="probability-chart" role="img" aria-label="Bar chart of Low, Moderate, and High probabilities">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probabilityData} layout="vertical" margin={{ top: 7, right: 18, left: 8, bottom: 3 }}>
                <CartesianGrid horizontal={false} stroke="#E4ECEF" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fontSize: 10, fill: '#64808B' }} />
                <YAxis type="category" dataKey="name" width={78} tick={{ fontSize: 11, fill: '#365B69' }} />
                <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, 'Probability']} />
                <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={25}>
                  {probabilityData.map((entry) => <Cell key={entry.name} fill={riskColors[entry.name as RiskClass]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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