// Version: 8.3.2
import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 240000,
})

function unwrap(error, fallback) {
  const message = error?.response?.data?.detail || error?.response?.data?.error || error?.message
  throw new Error(message || fallback)
}

export async function getStatus() {
  try {
    const { data } = await api.get('/', { timeout: 10000 })
    return data
  } catch (error) {
    unwrap(error, 'Unable to reach the backend status endpoint.')
  }
}

export async function analyzeAll(text) {
  try {
    const { data } = await api.get('/analyze-combined', {
      params: { text },
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
