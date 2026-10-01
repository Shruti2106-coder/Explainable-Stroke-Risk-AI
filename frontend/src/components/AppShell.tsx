import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Activity,
  BookOpenText,
  ChartNoAxesCombined,
  ClipboardList,
  HeartPulse,
  House,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { useAppContext } from '../context/AppContext'
import { Disclaimer } from './Primitives'

const navigation = [
  { to: '/', label: 'Home', icon: House, end: true },
  { to: '/assessment', label: 'Risk assessment', icon: ClipboardList },
  { to: '/results', label: 'Results', icon: Activity },
  { to: '/explanation', label: 'AI explanation', icon: Sparkles },
  { to: '/performance', label: 'Model performance', icon: ChartNoAxesCombined },
  { to: '/about', label: 'About & method', icon: BookOpenText },
]

const titles: Record<string, string> = {
  '/': 'Overview',
  '/assessment': 'Risk assessment',
  '/results': 'Assessment results',
  '/explanation': 'Explainable AI',
  '/performance': 'Model performance',
  '/about': 'About this project',
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const { serviceReady, modelInfoError, modelInfoLoading } = useAppContext()
  const serviceLabel = serviceReady ? 'Model ready' : modelInfoLoading ? 'Connecting to API' : 'Service unavailable'

  return (
    <div className="app-shell">
      {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-lockup">
          <span className="brand-mark"><HeartPulse size={22} strokeWidth={2.1} /></span>
          <span><strong>Stroke Insight</strong><small>Explainable assessment</small></span>
          <button className="icon-button sidebar-close" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={19} /></button>
        </div>
        <div className="sidebar-label">Workspace</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : ''}`}>
              <Icon size={18} strokeWidth={1.9} />
              <span>{label}</span>
              {label === 'Results' && location.pathname === '/results' && <span className="nav-current-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-safety">
          <span className="safety-icon"><ShieldCheck size={18} /></span>
          <div><strong>Education only</strong><p>Not for diagnosis or treatment decisions.</p></div>
        </div>
        <div className="sidebar-version">Stroke Insight <span>Module 10</span></div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
          <div className="topbar-page-title"><span className="topbar-kicker">Learning workspace</span><strong>{titles[location.pathname] ?? 'Stroke Insight'}</strong></div>
          <div className={`service-status ${serviceReady ? 'service-online' : modelInfoLoading ? 'service-loading' : 'service-offline'}`} title={modelInfoError ?? (serviceReady ? 'Saved model loaded' : serviceLabel)}>
            <span className="status-dot" />
            <span>{serviceLabel}</span>
          </div>
        </header>
        {modelInfoError && <div className="global-api-alert">Backend unavailable: {modelInfoError}</div>}
        <main className="page-container"><Outlet /></main>
        <footer className="app-footer"><Disclaimer compact /><span>Model output describes patterns in the supplied data only.</span></footer>
      </div>
    </div>
  )
}