import { Activity, ArrowRight, BrainCircuit, ChartNoAxesCombined, ShieldCheck, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'
import { Disclaimer, ErrorNotice, MetricCard, PageHeading } from '../components/Primitives'

export function HomePage() {
  const { modelInfo, modelInfoError, modelInfoLoading, serviceReady } = useAppContext()

  return (
    <>
      <PageHeading eyebrow="Educational assessment workspace" title="A clearer view of the model" description="Explore an educational stroke-risk estimate and see how the saved model reached its output." />
      {modelInfoError && <div className="mb-4"><ErrorNotice message={modelInfoError} /></div>}

      <section className="hero" aria-labelledby="home-title">
        <div className="hero-copy">
          <p className="eyebrow"><Sparkles size={14} /> Explainable machine learning</p>
          <h1 id="home-title">Understand Your Stroke Risk</h1>
          <p>Enter health information to view the finalized model’s estimate, class probabilities, and feature-level SHAP explanation. The result reflects model behavior, not a clinical assessment.</p>
          <div className="hero-actions">
            <Link to="/assessment" className="button button-primary">Assess your risk <ArrowRight size={16} /></Link>
            <Link to="/about" className="link-arrow">How this works <ArrowRight size={14} /></Link>
          </div>
          <div className="hero-trust-row">
            <span><ShieldCheck size={15} /> Educational only</span>
            <span><Activity size={15} /> {serviceReady ? 'Saved model connected' : modelInfoLoading ? 'Connecting to saved model' : 'Connect backend to assess'}</span>
          </div>
        </div>
        <div className="hero-visual">
          <img src="/assets/global-shap-importance.png" alt="Global SHAP feature-importance chart generated from the finalized model" />
          <div className="visual-caption"><Sparkles size={13} /> Model explanation · validation sample</div>
        </div>
      </section>

      <section className="home-section">
        <div className="section-title">
          <div><h2>Built around a transparent workflow</h2><p>From patient information to an inspectable model output.</p></div>
          <Link to="/performance" className="link-arrow">View model results <ArrowRight size={14} /></Link>
        </div>
        <div className="feature-grid">
          <article className="feature-card">
            <span className="feature-icon"><Activity size={19} /></span>
            <div><h3>AI Risk Assessment</h3><p>Three-class prediction from the approved health features, with probabilities for Low, Moderate, and High.</p></div>
          </article>
          <article className="feature-card">
            <span className="feature-icon"><BrainCircuit size={19} /></span>
            <div><h3>Explainable Predictions</h3><p>Local SHAP contributions are grouped back to meaningful source fields and encoded categories.</p></div>
          </article>
          <article className="feature-card">
            <span className="feature-icon"><ChartNoAxesCombined size={19} /></span>
            <div><h3>Health Insights</h3><p>Review the model’s validation comparisons and final held-out test metrics without treating them as medical guidance.</p></div>
          </article>
        </div>
      </section>

      <section className="home-section">
        <div className="section-title"><div><h2>Model snapshot</h2><p>Values from the saved training metadata.</p></div></div>
        {modelInfo ? (
          <div className="metric-grid">
            <MetricCard label="Selected model" value={modelInfo.model_name} detail="Finalized pipeline" icon={<BrainCircuit size={15} />} />
            <MetricCard label="Approved features" value={String(modelInfo.approved_features.length)} detail="Leakage fields excluded" icon={<ShieldCheck size={15} />} />
            <MetricCard label="Fit records" value={modelInfo.fit_rows.toLocaleString()} detail="Training + validation" icon={<Activity size={15} />} />
            <MetricCard label="Held-out macro F1" value={modelInfo.final_test_metrics.macro_f1.toFixed(3)} detail="Final test · one evaluation" icon={<ChartNoAxesCombined size={15} />} />
          </div>
        ) : (
          <div className="state-panel">Model information appears when the backend is available.</div>
        )}
      </section>

      <section className="home-section home-bottom-strip">
        <div><strong>Important context</strong><p>This project is for education and model-behavior exploration only.</p></div>
        <Disclaimer compact />
      </section>
    </>
  )
}