import type { GlobalExplanation, LocalExplanation, ModelInfo, PatientPayload, PredictionResult } from './types'

const configuredApiBase = import.meta.env.VITE_API_BASE_URL
const API_BASE = (
  configuredApiBase
    ? (/^https?:\/\//i.test(configuredApiBase) ? configuredApiBase : `https://${configuredApiBase}`)
    : 'http://127.0.0.1:8000'
).replace(/\/$/, '')

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch {
    throw new ApiError('The assessment service is unreachable. Check that the backend is running.', 0)
  }

  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    const message = Array.isArray(detail)
      ? detail.map((issue: { loc?: (string | number)[]; msg?: string }) => {
          const location = issue.loc?.[issue.loc.length - 1] ?? 'Input'
          return `${location}: ${issue.msg ?? 'Invalid value'}`
        }).join(' · ')
      : typeof detail === 'string'
        ? detail
        : `The request failed (${response.status}).`
    throw new ApiError(message, response.status)
  }
  return body as T
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>('/health'),
  modelInfo: () => request<ModelInfo>('/model-info'),
  predict: (patient: PatientPayload) => request<PredictionResult>('/predict', {
    method: 'POST',
    body: JSON.stringify(patient),
  }),
  explain: (patient: PatientPayload) => request<LocalExplanation>('/explain', {
    method: 'POST',
    body: JSON.stringify(patient),
  }),
  globalExplanation: () => request<GlobalExplanation>('/explanations/global'),
}

export { API_BASE }