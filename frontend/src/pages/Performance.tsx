import { Fragment } from 'react'
import { Activity, ChartNoAxesCombined, ShieldCheck, Target } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Disclaimer, ErrorNotice, MetricCard, PageHeading, Panel } from '../components/Primitives'
import { useAppContext } from '../context/AppContext'
import type { ModelComparisonRow } from '../types'

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

function comparisonData(rows: ModelComparisonRow[]) {
  const models = [...new Set(rows.map((row) => row.model_key))]
  return models.map((key) => {
    const baseline = rows.find((row) => row.model_key === key && row.stage === 'baseline_validation')
    const tuned = rows.find((row) => row.model_key === key && row.stage === 'tuned_validation')
    return { model: tuned?.model ?? baseline?.model ?? key, baseline: baseline?.macro_f1 ?? null, tuned: tuned?.macro_f1 ?? null }
  })
}

export function ModelPerformancePage() {
  const { modelInfo, modelInfoError } = useAppContext()
  if (!modelInfo) return <><PageHeading eyebrow="Model performance" title="Evaluation results" description="Actual metrics from the saved final model metadata." />{modelInfoError ? <ErrorNotice message={modelInfoError} /> : <div className="state-panel">Loading saved evaluation metrics.</div>}</>

  const metrics = modelInfo.final_test_metrics
  const order = modelInfo.final_test_class_order
  const matrix = metrics.confusion_matrix
  const maximum = Math.max(...matrix.flat())
  const comparison = comparisonData(modelInfo.model_comparison_validation)

  return (
    <>
      <PageHeading eyebrow="Model performance" title="Evaluation results" description={`${modelInfo.model_name} · final untouched test set. Macro metrics weight each class equally.`} action={<span className="evaluation-tag"><ShieldCheck size={14} /> Held-out test</span>} />
      <div className="metric-grid">
        <MetricCard label="Accuracy" value={percent(metrics.accuracy)} detail="Overall correct predictions" icon={<Target size={15} />} />
        <MetricCard label="Macro precision" value={percent(metrics.macro_precision)} detail="Equal class weighting" icon={<ChartNoAxesCombined size={15} />} />
        <MetricCard label="Macro recall" value={percent(metrics.macro_recall)} detail="Equal class weighting" icon={<Activity size={15} />} />
        <MetricCard label="Macro F1" value={percent(metrics.macro_f1)} detail="Equal class weighting" icon={<ChartNoAxesCombined size={15} />} />
      </div>

      <section className="home-section">
        <div className="section-title"><div><h2>Additional final-test metrics</h2><p>Actual values from the saved one-time final evaluation.</p></div></div>
        <div className="metric-grid metric-grid-three">
          <MetricCard label="Weighted precision" value={percent(metrics.weighted_precision)} />
          <MetricCard label="Weighted recall" value={percent(metrics.weighted_recall)} />
          <MetricCard label="Weighted F1" value={percent(metrics.weighted_f1)} />
          <MetricCard label="Macro OVR ROC-AUC" value={percent(metrics.roc_auc_ovr_macro)} />
          <MetricCard label="Macro OVR PR-AUC" value={percent(metrics.pr_auc_ovr_macro_average_precision)} detail="Mean per-class average precision" />
          <MetricCard label="Fitted rows" value={modelInfo.fit_rows.toLocaleString()} detail="Training + validation" />
        </div>
      </section>

      <section className="home-section page-grid">
        <div className="col-5">
          <div className="section-title"><div><h2>Confusion matrix</h2><p>Rows are actual; columns are predicted.</p></div></div>
          <Panel>
            <div className="confusion-grid" role="table" aria-label="Final test confusion matrix">
              <div className="confusion-axis" />
              {order.map((risk) => <div className="confusion-col-label" role="columnheader" key={`head-${risk}`}>{risk}</div>)}
              {matrix.map((row, rowIndex) => (
                <Fragment key={`row-${order[rowIndex]}`}>
                  <div className="confusion-axis" role="rowheader">{order[rowIndex]}</div>
                  {row.map((value, columnIndex) => {
                    const intensity = Math.min(.58, .08 + (value / maximum) * .5)
                    return <div className="confusion-cell" role="cell" key={`${rowIndex}-${columnIndex}`} style={{ backgroundColor: `rgba(95,159,181,${intensity})` }}>{value.toLocaleString()}</div>
                  })}
                </Fragment>
              ))}
            </div>
            <p className="confusion-legend">Test records: {metrics.classification_report['macro avg'].support?.toLocaleString()}. The test set was not used for tuning.</p>
          </Panel>
        </div>
        <div className="col-7">
          <div className="section-title"><div><h2>Baseline vs tuned</h2><p>Validation macro-F1 used during model selection.</p></div></div>
          <Panel>
            {comparison.length ? (
              <div className="comparison-chart" role="img" aria-label="Baseline and tuned validation macro-F1 comparison">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparison} margin={{ top: 12, right: 10, left: 0, bottom: 32 }}>
                    <CartesianGrid vertical={false} stroke="#E8EFF1" />
                    <XAxis dataKey="model" interval={0} angle={-10} textAnchor="end" height={55} tick={{ fontSize: 9, fill: '#64808B' }} />
                    <YAxis domain={[0, 1]} tickFormatter={percent} tick={{ fontSize: 9, fill: '#64808B' }} />
                    <Tooltip formatter={(value) => [percent(Number(value)), 'Validation macro-F1']} />
                    <Bar dataKey="baseline" name="Baseline" fill="#A9BAC0" radius={[4, 4, 0, 0]} barSize={23} />
                    <Bar dataKey="tuned" name="Tuned" fill="#5F9FB5" radius={[4, 4, 0, 0]} barSize={23}>
                      {comparison.map((row) => <Cell key={row.model} fill={row.model === modelInfo.model_name ? '#4F8F98' : '#5F9FB5'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : <div className="state-panel">Comparison metrics are not available.</div>}
            <p className="comparison-footnote">Final-model choice was made using validation macro-F1 with minority-class recall and OVR metrics as tie-breakers. Test metrics are shown separately above.</p>
          </Panel>
        </div>
      </section>
      <div className="home-section"><Disclaimer /></div>
    </>
  )
}