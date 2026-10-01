import { useEffect, useState } from 'react'
import { ArrowDownRight, ArrowUpRight, Info, Sparkles } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, ApiError } from '../api'
import { Disclaimer, EmptyState, ErrorNotice, LoadingState, PageHeading, Panel, RiskBadge } from '../components/Primitives'
import { useAppContext } from '../context/AppContext'
import type { GlobalExplanation, LocalContribution, LocalExplanation } from '../types'

function ContributionChart({ title, items, positive }: { title: string; items: LocalContribution[]; positive: boolean }) {
  const chartData = items.slice(0, 8).map((entry) => ({
    feature: entry.feature,
    value: entry.shap_value,
  }))
  return (
    <Panel>
      <div className="section-title"><div className="contribution-section-title mt-0"><span className={positive ? 'dot-positive' : 'dot-negative'} />{title}</div></div>
      {chartData.length ? (
        <div className="explanation-chart" role="img" aria-label={title}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 12, left: 8, bottom: 4 }}>
              <CartesianGrid horizontal={false} stroke="#E8EFF1" />
              <XAxis type="number" tick={{ fontSize: 9, fill: '#64808B' }} />
              <YAxis type="category" dataKey="feature" width={145} tick={{ fontSize: 9, fill: '#365B69' }} />
              <ReferenceLine x={0} stroke="#9CB0B7" />
              <Tooltip formatter={(value) => [Number(value).toFixed(3), 'SHAP margin']} />
              <Bar dataKey="value" radius={3} barSize={17}>
                {chartData.map((entry) => <Cell key={entry.feature} fill={entry.value >= 0 ? '#D96B6B' : '#4F8F98'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : <div className="state-panel">No non-zero contributions for this side.</div>}
    </Panel>
  )
}

export function ExplanationPage() {
  const { patient, prediction } = useAppContext()
  const [local, setLocal] = useState<LocalExplanation | null>(null)
  const [global, setGlobal] = useState<GlobalExplanation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!patient) return
    let active = true
    setLoading(true)
    setError(null)
    Promise.all([api.explain(patient), api.globalExplanation()])
      .then(([localResult, globalResult]) => {
        if (!active) return
        setLocal(localResult)
        setGlobal(globalResult)
      })
      .catch((reason: Error) => {
        if (active) setError(reason instanceof ApiError ? reason.message : 'The explanation could not be loaded.')
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [patient])

  if (!patient || !prediction) {
    return <><PageHeading eyebrow="Explainable AI" title="How the model arrived here" description="Local SHAP is available after an assessment." /><EmptyState title="An assessment is needed" description="Submit patient information first. Then you can inspect the class-specific model contributions." /></>
  }

  const globalData = global?.feature_importance.slice(0, 12).map((item) => ({ feature: item.feature, importance: item.mean_abs_shap })) ?? []

  return (
    <>
      <PageHeading eyebrow="Explainable AI" title="How the model arrived here" description="SHAP describes the model’s output pattern for this input and across a validation sample. It does not identify medical causes." action={<RiskBadge risk={prediction.predicted_risk} />} />
      {error && <div className="mb-4"><ErrorNotice message={error} /></div>}
      {loading && <div className="mb-4"><LoadingState label="Calculating class-specific SHAP values" /></div>}
      {local && (
        <>
          <div className="notice notice-info mb-4"><Info size={18} /><div><strong>Reading this explanation</strong><p>{local.contribution_note} The output unit is not a probability change.</p></div></div>
          <section className="explanation-top-grid">
            <ContributionChart title={`Features moving toward ${local.predicted_risk_class}`} items={local.features_contributing_toward_prediction} positive />
            <ContributionChart title="Features moving away" items={local.features_contributing_away_from_prediction} positive={false} />
          </section>
          <section className="home-section">
            <div className="section-title"><div><h2>Model-wide feature importance</h2><p>Mean absolute SHAP across three class outputs, grouped back to original input fields.</p></div><span className="eyebrow">{global?.rows_explained ?? '—'} validation records</span></div>
            {global ? (
              <Panel>
                <div className="global-ranking-chart" role="img" aria-label="Global SHAP feature importance bar chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={globalData} layout="vertical" margin={{ top: 5, right: 18, left: 12, bottom: 5 }}>
                      <CartesianGrid horizontal={false} stroke="#E8EFF1" />
                      <XAxis type="number" tick={{ fontSize: 9, fill: '#64808B' }} />
                      <YAxis type="category" dataKey="feature" width={190} tick={{ fontSize: 10, fill: '#365B69' }} />
                      <Tooltip formatter={(value) => [Number(value).toFixed(3), 'Mean |SHAP|']} />
                      <Bar dataKey="importance" fill="#5F9FB5" radius={[0, 4, 4, 0]} barSize={17} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="comparison-footnote">Global rankings summarize model attribution in this validation sample; they are not medical causes or treatment recommendations.</p>
              </Panel>
            ) : <div className="state-panel">Global explanation is loading.</div>}
          </section>
          <section className="home-section">
            <div className="section-title"><div><h2>What the model used</h2><p>Encoded categories are combined into their source field for the primary explanation.</p></div><span><Sparkles size={14} className="inline mr-1" />{local.explanation_output.replaceAll('_', ' ')}</span></div>
            <Panel className="explain-how-panel">
              <ul className="explain-how-list">
                <li><ArrowUpRight size={16} /> Positive SHAP values raise the selected class’s raw model output relative to its baseline.</li>
                <li><ArrowDownRight size={16} /> Negative SHAP values lower that class output. Other class outputs are separate model values.</li>
                <li><Info size={16} /> One-hot category details are mapped to labels such as “Smoking_Status = Former”; the grouped list reports the source field.</li>
              </ul>
            </Panel>
          </section>
          <div className="mt-5"><Disclaimer /></div>
        </>
      )}
    </>
  )
}