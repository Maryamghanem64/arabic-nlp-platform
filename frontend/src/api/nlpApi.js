import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export async function analyzeAll(text, timeout = 240000) {
  const { data } = await axios.get(`${API}/analyze-combined`, {
    params: { text },
    timeout,
  })
  return data
}

export async function evaluateText(text, timeout = 240000) {
  const { data } = await axios.get(`${API}/evaluate?text=${encodeURIComponent(text)}`, {
    timeout,
  })
  return data
}

export async function fusionText(text, timeout = 240000) {
  const { data } = await axios.get(`${API}/fusion`, {
    params: { text },
    timeout,
  })
  return data
}
