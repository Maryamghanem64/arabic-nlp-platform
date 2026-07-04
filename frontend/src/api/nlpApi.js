// Version: 8.3.2
import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
export const HEALTH_TIMEOUT_MS = 5000
export const DASHBOARD_TIMEOUT_MS = 5000
export const ANALYSIS_TIMEOUT_MS = 120000

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: ANALYSIS_TIMEOUT_MS,
})

function unwrap(error, fallback) {
  const message = error?.response?.data?.detail || error?.response?.data?.error || error?.message
  throw new Error(message || fallback)
}

export async function getToolsStatus() {
  try {
    const { data } = await api.get('/tools/status', { timeout: HEALTH_TIMEOUT_MS })
    return data
  } catch (error) {
    unwrap(error, 'Unable to reach backend tools status endpoint.')
  }
}

export async function getStatus() {
  // Backwards compatibility: existing pages still call GET /health.
  try {
    const { data } = await api.get('/health', { timeout: HEALTH_TIMEOUT_MS })
    return data
  } catch (error) {
    unwrap(error, 'Unable to reach the backend health endpoint.')
  }
}

export async function preloadSinaTools() {
  const { data } = await api.post('/tools/sinatools/preload', {}, { timeout: HEALTH_TIMEOUT_MS })
  return data
}


export async function getDemoToolHealth(runSample = false) {
  try {
    const { data } = await api.get('/health/demo-tools', {
      params: { run_sample: runSample },
      timeout: runSample ? ANALYSIS_TIMEOUT_MS : DASHBOARD_TIMEOUT_MS,
    })
    return data
  } catch (error) {
    unwrap(error, 'Unable to load dashboard tool health.')
  }
}

export async function analyzeAll(text) {
  try {
    const { data } = await api.get('/analyze-combined', {
      params: { text },
      timeout: ANALYSIS_TIMEOUT_MS,
    })
    return data
  } catch (error) {
    unwrap(error, 'Unable to run combined analysis.')
  }
}

export async function analyzeTool(tool, text) {
  try {
    const { data } = await api.get(`/analyze/${tool}`, {
      params: { text },
      timeout: ANALYSIS_TIMEOUT_MS,
    })
    return data
  } catch (error) {
    unwrap(error, `Unable to run ${tool} analysis.`)
  }
}

export async function evaluateText(text) {
  try {
    const { data } = await api.get('/evaluate', {
      params: { text },
      timeout: ANALYSIS_TIMEOUT_MS,
    })
    return data
  } catch (error) {
    unwrap(error, 'Unable to evaluate tool agreement.')
  }
}

export async function fusionText(text) {
  try {
    const { data } = await api.get('/fusion', {
      params: { text },
      timeout: ANALYSIS_TIMEOUT_MS,
    })
    return data
  } catch (error) {
    unwrap(error, 'Unable to run fusion.')
  }
}

export function exportUrl(text, format = 'json') {
  const params = new URLSearchParams({ text, format })
  return `${API_BASE_URL}/export?${params.toString()}`
}
