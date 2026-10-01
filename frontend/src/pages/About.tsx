import { CheckCircle2, Database, GitBranch, ShieldAlert, Sparkles } from 'lucide-react'
import { PageHeading, Panel } from '../components/Primitives'
import { useAppContext } from '../context/AppContext'

export function AboutPage() {
  const { modelInfo } = useAppContext()
  return (
    <>
      <PageHeading eyebrow="About & methodology" title="An educational model, made inspectable" description="This workspace documents a reproducible machine-learning workflow using the supplied dataset." />
      <div className="about-grid">
        <div className="about-copy">
          <Panel>
            <h2>Project objective</h2>
            <p>Stroke Insight is a full-stack educational project for exploring a three-class stroke-risk prediction pipeline. It presents model output alongside probabilities, saved evaluation metrics, and local/global SHAP explanations.</p>
            <h2>Dataset & feature policy</h2>
            <p>The project uses the supplied 50,000-row, 40-column dataset. The persisted feature configuration approves {modelInfo?.approved_features.length ?? '—'} prediction inputs and keeps <code>Stroke_Risk</code> as the target. Patient identifiers and target-derived/downstream columns are excluded.</p>
            <h2>Machine-learning workflow</h2>
            <ul className="method-list">
              <li><CheckCircle2 size={15} /> Data quality, missingness, category domains, and possible target leakage were analyzed before modeling.</li>
              <li><CheckCircle2 size={15} /> Preprocessing is included in the scikit-learn pipeline and is fitted on training partitions, not the complete dataset.</li>
              <li><CheckCircle2 size={15} /> Models were compared with stratified validation; the final test set was held for one final evaluation.</li>
              <li><CheckCircle2 size={15} /> SHAP explanations summarize model-output contributions and are not causal explanations.</li>
            </ul>
            <h2 className="mt-6">Limitations</h2>
            <p>Input bounds reflect the values observed in the supplied training data, not clinical reference intervals. Performance is specific to these prepared splits and this dataset. The model may not generalize to other populations or workflows; probabilities should not guide care.</p>
          </Panel>
        </div>
        <aside className="about-side">
          <Panel>
            <div className="about-stat"><span>Target</span><strong>Stroke_Risk</strong></div>
            <div className="about-stat"><span>Classes</span><strong>Low · Moderate · High</strong></div>
            <div className="about-stat"><span>Approved inputs</span><strong>{modelInfo?.approved_features.length ?? '—'} features</strong></div>
            <div className="about-stat"><span>Final model</span><strong>{modelInfo?.model_name ?? 'Loading'}</strong></div>
            <div className="about-stat"><span>Explanation method</span><strong>SHAP TreeExplainer</strong></div>
          </Panel>
          <div className="prominent-disclaimer mt-4"><strong><ShieldAlert size={16} className="inline mr-1" />Important disclaimer</strong>This application is an educational machine-learning project and is not a medical diagnosis or a substitute for professional medical advice.</div>
          <p className="muted-copy mt-4"><Database size={14} className="inline mr-1" />Predictions and explanations are generated from the finalized saved pipeline.</p>
          <p className="muted-copy"><GitBranch size={14} className="inline mr-1" /><Sparkles size={14} className="inline mr-1" />The frontend does not train or alter the model.</p>
        </aside>
      </div>
    </>
  )
}