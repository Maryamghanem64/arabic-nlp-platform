import { computed, onMounted, ref } from 'vue'
import { getToolsStatus, getStatus } from '../api/nlpApi'

import { TOOL_KEYS } from '../config/tools'
import { statusGroupsFromMap } from '../utils/researchSemantics'

const HEAVY_LAZY_TOOLS = new Set(['stanza', 'arabert', 'sinatools'])

export function useToolStatus() {
  const toolStatuses = ref({})
  const loading = ref(true)
  const error = ref(null)

  const groupedStatuses = computed(() => statusGroupsFromMap(toolStatuses.value))
  const activeTools = computed(() => groupedStatuses.value.activeTools)
  const partialTools = computed(() => groupedStatuses.value.partialTools)
  const lazyTools = computed(() => groupedStatuses.value.lazyTools)
  const loadingTools = computed(() => groupedStatuses.value.loadingTools)
  const excludedTools = computed(() => groupedStatuses.value.excludedTools)
  const unavailableTools = computed(() => groupedStatuses.value.unavailableTools)
  const errorTools = computed(() => groupedStatuses.value.errorTools)
  const degradedTools = computed(() => groupedStatuses.value.degradedTools)

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
    groupedStatuses,
    activeTools,
    partialTools,
    lazyTools,
    loadingTools,
    excludedTools,
    unavailableTools,
    errorTools,
    degradedTools,
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
