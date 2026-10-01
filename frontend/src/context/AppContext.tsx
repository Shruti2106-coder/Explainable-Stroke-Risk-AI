import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api } from '../api'
import type { ModelInfo, PatientPayload, PredictionResult } from '../types'

interface AppContextValue {
  modelInfo: ModelInfo | null
  modelInfoError: string | null
  modelInfoLoading: boolean
  serviceReady: boolean
  patient: PatientPayload | null
  prediction: PredictionResult | null
  saveAssessment: (patient: PatientPayload, prediction: PredictionResult) => void
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelInfoError, setModelInfoError] = useState<string | null>(null)
  const [modelInfoLoading, setModelInfoLoading] = useState(true)
  const [serviceReady, setServiceReady] = useState(false)
  const [patient, setPatient] = useState<PatientPayload | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)

  useEffect(() => {
    let active = true
    Promise.all([api.health(), api.modelInfo()])
      .then(([health, info]) => {
        if (!active) return
        setServiceReady(health.status === 'ok' && health.model_loaded)
        setModelInfo(info)
        setModelInfoError(null)
      })
      .catch((error: Error) => {
        if (!active) return
        setServiceReady(false)
        setModelInfoError(error.message)
      })
      .finally(() => { if (active) setModelInfoLoading(false) })
    return () => { active = false }
  }, [])

  const saveAssessment = (nextPatient: PatientPayload, nextPrediction: PredictionResult) => {
    setPatient(nextPatient)
    setPrediction(nextPrediction)
  }

  return (
    <AppContext.Provider value={{ modelInfo, modelInfoError, modelInfoLoading, serviceReady, patient, prediction, saveAssessment }}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppContext() {
  const context = useContext(AppContext)
  if (!context) throw new Error('useAppContext must be used inside AppProvider')
  return context
}