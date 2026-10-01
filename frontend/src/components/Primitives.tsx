import type { ReactNode } from 'react'
import { AlertCircle, ArrowRight, LoaderCircle, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'

export function PageHeading({ eyebrow, title, description, action }: {
  eyebrow?: string
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {action && <div className="page-heading-action">{action}</div>}
    </div>
  )
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function MetricCard({ label, value, detail, icon }: { label: string; value: string; detail?: string; icon?: ReactNode }) {
  return (
    <div className="metric-card">
      <div className="metric-card-top"><span>{label}</span>{icon && <span className="metric-icon">{icon}</span>}</div>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  )
}

export function RiskBadge({ risk, large = false }: { risk: string; large?: boolean }) {
  const colorClass = risk.toLowerCase()
  return <span className={`risk-badge risk-${colorClass} ${large ? 'risk-badge-large' : ''}`}>{risk}</span>
}

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return <div className="state-panel"><LoaderCircle className="animate-spin" size={22} /><span>{label}</span></div>
}

export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="notice notice-error" role="alert">
      <AlertCircle size={19} aria-hidden="true" />
      <div><strong>Something needs attention</strong><p>{message}</p></div>
      {onRetry && <button className="text-button" onClick={onRetry}>Retry</button>}
    </div>
  )
}

export function EmptyState({ title, description, actionLabel = 'Start an assessment' }: { title: string; description: string; actionLabel?: string }) {
  return (
    <Panel className="empty-state">
      <div className="empty-symbol"><ShieldCheck size={24} /></div>
      <h2>{title}</h2>
      <p>{description}</p>
      <Link to="/assessment" className="button button-primary">{actionLabel}<ArrowRight size={17} /></Link>
    </Panel>
  )
}

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`disclaimer ${compact ? 'disclaimer-compact' : ''}`}>
      <ShieldCheck size={17} aria-hidden="true" />
      <span>This educational machine-learning project is not a medical diagnosis or a substitute for professional medical advice.</span>
    </div>
  )
}