import { computed, onMounted, ref } from 'vue'
import axios from 'axios'
import { API_BASE_URL } from '../api/nlpApi'
import { TOOL_KEYS } from '../config/tools'

export function useToolStatus() {
  const toolStatuses = ref({})
  const loading = ref(true)
  const error = ref(null)

  const activeTools = computed(() =>
    TOOL_KEYS.filter((tool) => toolStatuses.value[tool]?.status === 'ok'),
  )

  function normalizeStatuses(payload) {
    const raw = payload?.tools || payload?.statuses || payload || {}
    return Object.fromEntries(
      TOOL_KEYS.map((tool) => [tool, normalizeStatus(raw?.[tool])]),
    )
  }

  async function refresh() {
    loading.value = true
    error.value = null

    try {
      const { data } = await axios.get(`${API_BASE_URL}/`, { timeout: 12000 })
      toolStatuses.value = normalizeStatuses(data)
    } catch (e) {
      toolStatuses.value = Object.fromEntries(
        TOOL_KEYS.map((tool) => [tool, { status: 'unknown' }]),
      )
      error.value = e
    } finally {
      loading.value = false
    }
  }

  function toolStatus(key) {
    return toolStatuses.value[key]?.status || 'unknown'
  }

  function toolReason(key) {
    return toolStatuses.value[key]?.reason || toolStatuses.value[key]?.error || ''
  }

  onMounted(refresh)

  return {
    toolStatuses,
    activeTools,
    loading,
    error,
    refresh,
    toolStatus,
    toolReason,
  }
}

function normalizeStatus(entry) {
  if (!entry) return { status: 'unknown' }
  if (typeof entry === 'string') return { status: entry.toLowerCase() }
  return {
    ...entry,
    status: String(entry.status || 'unknown').toLowerCase(),
  }
}
