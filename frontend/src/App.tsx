import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppProvider } from './context/AppContext'
import { AppShell } from './components/AppShell'
import { LoadingState } from './components/Primitives'

const HomePage = lazy(() => import('./pages/Home').then((page) => ({ default: page.HomePage })))
const RiskAssessmentPage = lazy(() => import('./pages/RiskAssessment').then((page) => ({ default: page.RiskAssessmentPage })))
const ResultsPage = lazy(() => import('./pages/Results').then((page) => ({ default: page.ResultsPage })))
const ExplanationPage = lazy(() => import('./pages/Explanation').then((page) => ({ default: page.ExplanationPage })))
const ModelPerformancePage = lazy(() => import('./pages/Performance').then((page) => ({ default: page.ModelPerformancePage })))
const AboutPage = lazy(() => import('./pages/About').then((page) => ({ default: page.AboutPage })))

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Suspense fallback={<div className="page-container"><LoadingState label="Opening workspace" /></div>}>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<HomePage />} />
              <Route path="/assessment" element={<RiskAssessmentPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/explanation" element={<ExplanationPage />} />
              <Route path="/performance" element={<ModelPerformancePage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AppProvider>
  )
}