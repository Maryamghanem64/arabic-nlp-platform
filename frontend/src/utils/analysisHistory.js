const STORAGE_KEY = 'arabic-nlp-analysis-history'
const MAX_ITEMS = 8

export function readAnalysisHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const items = raw ? JSON.parse(raw) : []
    return Array.isArray(items) ? items : []
  } catch {
    return []
  }
}

export function recordAnalysis(entry) {
  if (!entry || typeof window === 'undefined') return []

  const next = [
    {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      at: new Date().toISOString(),
      ...entry,
    },
    ...readAnalysisHistory().filter((item) => item?.text !== entry.text),
  ].slice(0, MAX_ITEMS)

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  window.dispatchEvent(new Event('analysis-history-updated'))
  return next
}
