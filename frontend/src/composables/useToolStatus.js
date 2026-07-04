import { computed, onMounted, ref } from 'vue'
import { getToolsStatus, getStatus } from '../api/nlpApi'

import { TOOL_KEYS } from '../config/tools'

const EXCLUDED_TOOLS = new Set(['madamira'])
const COUNTABLE_STATUSES = new Set(['ok', 'partial', 'lazy', 'loading'])
const HEAVY_LAZY_TOOLS = new Set(['stanza', 'arabert', 'sinatools'])

export function useToolStatus() {
  const toolStatuses = ref({})
  const loading = ref(true)
  const error = ref(null)

  const activeTools = computed(() =>
    TOOL_KEYS.filter((tool) => !EXCLUDED_TOOLS.has(tool) && COUNTABLE_STATUSES.has(toolStatuses.value[tool]?.status)),
  )

  function normalizeStatuses(payload) {
    const raw = payload?.tools || payload?.data?.tools || payload?.statuses || payload?.data?.statuses || payload || {}
    return Object.fromEntries(
      TOOL_KEYS.map((tool) => [tool, normalizeStatus(tool, raw?.[tool])]),
    )
  }

  async function refresh() {
    loading.value = true
    error.value = null

    try {
      const data = await getToolsStatus()
      toolStatuses.value = normalizeStatuses(data)
    } catch (e) {
      // Fallback to legacy health payload.
      try {
        const legacy = await getStatus()
        toolStatuses.value = normalizeStatuses(legacy)
      } catch {
        toolStatuses.value = Object.fromEntries(TOOL_KEYS.map((tool) => [tool, { status: 'unknown' }]))
      }
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

  function toolMeta(key) {
    return toolStatuses.value[key] || { status: 'unknown' }
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
    toolMeta,
  }
}

function normalizeStatus(tool, entry) {
  if (!entry) return { status: 'unknown' }
  if (typeof entry === 'string') return { status: entry.toLowerCase() }
  const rawStatus = String(entry.status || 'unknown').toLowerCase()
  const status = tool === 'madamira' && rawStatus !== 'ok'
    ? 'excluded'
    : HEAVY_LAZY_TOOLS.has(tool) && rawStatus === 'ok' && entry.loaded === false
      ? 'lazy'
      : rawStatus
  return {
    ...entry,
    status,
  }
}
