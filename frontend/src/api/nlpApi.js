// Version: 8.3.1
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
  // Tools like Stanza/Farasa can be slow; keep a long timeout.
  timeout: 240000,
})

export async function analyzeAll(text) {
  const { data } = await api.get('/analyze-combined', {
    params: { text },
  })
  return data
}

export async function evaluateText(text) {
  const { data } = await api.get('/evaluate', {
    params: { text },
  })
  return data
}

export async function fusionText(text) {
  const { data } = await api.get('/fusion', {
    params: { text },
  })
  return data
}

