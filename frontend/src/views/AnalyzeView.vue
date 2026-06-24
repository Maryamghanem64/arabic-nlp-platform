<template>
  <div class="page-wrap analyze-page page-stack">
    <section class="hero-band analyze-hero">
      <div class="hero-content">
        <span class="eyebrow">Single tool deep analysis</span>
        <h1 class="hero-title">Select any tool, or run the combined backend path.</h1>
        <p class="hero-copy">
          Tool availability comes from <code>GET /</code>. Selecting a single active tool calls
          <code>/analyze/{tool}</code>, while All tools calls <code>/analyze-combined</code>.
        </p>
      </div>
    </section>

    <section class="panel panel-pad">
      <div class="section-head">
        <div>
          <h2 class="section-title">Analysis Input</h2>
          <p class="section-subtitle">{{ tokenEstimate }} token{{ tokenEstimate === 1 ? '' : 's' }} estimated</p>
        </div>
        <div class="actions-row compact-actions">
          <button class="btn btn-subtle" @click="loadSample">Sample</button>
          <button class="btn btn-subtle" @click="clear">Clear</button>
        </div>
      </div>

      <textarea
        v-model="inputText"
        class="textarea arabic"
        dir="rtl"
        lang="ar"
        placeholder="اكتب النص العربي هنا..."
      ></textarea>

      <div class="run-row">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="analyze">
          {{ loading ? 'Analyzing...' : 'Run Analysis' }}
        </button>
        <button class="btn btn-secondary" :disabled="!hasResults" @click="copyCurrentJson">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

    <section class="panel panel-pad selector-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">Tool Selector</h2>
          <p class="section-subtitle">The list is driven by the shared tool config and live backend status.</p>
        </div>
      </div>

      <div class="selector-grid">
        <button
          class="selector-card all-tools"
          :class="{ active: selectedTool === 'all' }"
          type="button"
          @click="selectTool('all')"
        >
          <span class="selector-dot" :style="{ backgroundColor: '#0f172a' }"></span>
          <div class="selector-copy">
            <span class="selector-label">All tools</span>
            <span class="selector-subtitle">Combined analysis</span>
          </div>
        </button>

        <button
          v-for="tool in toolOptions"
          :key="tool.key"
          class="selector-card"
          :class="{ active: selectedTool === tool.key }"
          type="button"
          @click="selectTool(tool.key)"
        >
          <span class="selector-dot" :style="{ backgroundColor: tool.color }"></span>
          <div class="selector-copy">
            <div class="selector-row">
              <span class="selector-label">{{ tool.label }}</span>
              <span v-if="statusBadge(tool.key).show" :class="['pill', statusBadge(tool.key).className]">
                {{ statusBadge(tool.key).label }}
              </span>
            </div>
            <span class="selector-subtitle">{{ tool.type }}</span>
          </div>
        </button>
      </div>

      <div v-if="selectionNotice" class="selection-notice" :class="selectionNotice.kind">
        <strong>{{ selectionNotice.title }}</strong>
        <p>{{ selectionNotice.message }}</p>
        <div v-if="selectionNotice.pending" class="actions-row compact-actions">
          <button class="btn btn-secondary" @click="confirmPendingSelection">Use tool anyway</button>
          <button class="btn btn-subtle" @click="cancelPendingSelection">Cancel</button>
        </div>
      </div>
    </section>

    <div v-if="statusError && !toolStatusesLoaded" class="error-state">
      <div>
        <strong>Backend status unavailable</strong>
        <p>{{ statusError.message || 'Could not load tool availability from GET /.' }}</p>
      </div>
    </div>

    <div v-if="loading" class="loading-state analysis-loading">
      <div class="loading-stack">
        <span class="skeleton"></span>
        <span class="skeleton wide"></span>
        <span class="skeleton short"></span>
      </div>
    </div>

    <div v-if="error" class="error-state">
      <div>
        <strong>Analysis failed</strong>
        <p>{{ error }}</p>
        <button class="btn btn-secondary" @click="analyze">Retry</button>
      </div>
    </div>

    <template v-if="hasResults && !loading">
      <section class="panel panel-pad">
        <div class="analysis-tabs">
          <button v-for="tab in tabs" :key="tab.key" type="button" :class="{ active: activeTab === tab.key }" @click="activeTab = tab.key">
            {{ tab.label }}
          </button>
        </div>

        <div v-if="activeTab === 'results'" class="tab-panel">
          <div v-if="selectedTool === 'all'" class="results-grid">
            <article v-for="tool in toolOptions" :key="tool.key" class="tool-result-card">
              <div class="tool-result-head" :style="{ borderTopColor: tool.color }">
                <div class="tool-result-title">
                  <span class="tool-result-bar" :style="{ backgroundColor: tool.color }"></span>
                  <div>
                    <h3>{{ tool.label }}</h3>
                    <p>{{ tool.type }}</p>
                  </div>
                </div>
                <span :class="['pill', statusPill(toolStatus(tool.key))]">{{ statusLabel(toolStatus(tool.key)) }}</span>
              </div>

              <div v-if="toolStatus(tool.key) === 'ok' && toolRows(tool.key).length" class="table-scroll tool-table">
                <table>
                  <thead>
                    <tr>
                      <th>Token</th>
                      <th v-for="field in toolColumns(tool.key)" :key="`${tool.key}-${field}`">{{ fieldLabel(field) }}</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in toolRows(tool.key)" :key="`${tool.key}-${row.index}`">
                      <td class="arabic" dir="rtl" lang="ar">{{ row.surface }}</td>
                      <td v-for="field in toolColumns(tool.key)" :key="`${tool.key}-${row.index}-${field}`">
                        <span :class="cellClass(field, row.values[field])" :dir="isArabicField(field) ? 'rtl' : null" :lang="isArabicField(field) ? 'ar' : null">
                          {{ formatCellValue(field, row.values[field]) }}
                        </span>
                      </td>
                      <td>
                        <span :class="['pill', confidencePill(row.confidence)]">{{ row.confidence || '—' }}</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-else-if="toolStatus(tool.key) === 'ok'" class="empty-state">No token output is available for this tool.</div>
              <div v-else class="status-card">
                <strong>{{ statusLabel(toolStatus(tool.key)) }}</strong>
                <p>{{ toolReason(tool.key) || 'The backend returned a safe unavailable response.' }}</p>
              </div>
            </article>
          </div>

          <article v-else class="panel panel-pad single-result-card">
            <div class="tool-result-head" :style="{ borderTopColor: selectedToolMeta.color }">
              <div class="tool-result-title">
                <span class="tool-result-bar" :style="{ backgroundColor: selectedToolMeta.color }"></span>
                <div>
                  <h3>{{ selectedToolMeta.label }}</h3>
                  <p>{{ selectedToolMeta.type }}</p>
                </div>
              </div>
              <span :class="['pill', statusPill(toolStatus(selectedTool))]">{{ statusLabel(toolStatus(selectedTool)) }}</span>
            </div>

            <div v-if="toolStatus(selectedTool) === 'ok' && toolRows(selectedTool).length" class="table-scroll tool-table">
              <table>
                <thead>
                  <tr>
                    <th>Token</th>
                    <th v-for="field in toolColumns(selectedTool)" :key="`${selectedTool}-${field}`">{{ fieldLabel(field) }}</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in toolRows(selectedTool)" :key="`${selectedTool}-${row.index}`">
                    <td class="arabic" dir="rtl" lang="ar">{{ row.surface }}</td>
                    <td v-for="field in toolColumns(selectedTool)" :key="`${selectedTool}-${row.index}-${field}`">
                      <span :class="cellClass(field, row.values[field])" :dir="isArabicField(field) ? 'rtl' : null" :lang="isArabicField(field) ? 'ar' : null">
                        {{ formatCellValue(field, row.values[field]) }}
                      </span>
                    </td>
                    <td>
                      <span :class="['pill', confidencePill(row.confidence)]">{{ row.confidence || '—' }}</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="status-card">
              <strong>{{ statusLabel(toolStatus(selectedTool)) }}</strong>
              <p>{{ toolReason(selectedTool) || 'The backend returned a safe unavailable response.' }}</p>
            </div>
          </article>
        </div>

        <div v-if="activeTab === 'fusion' && selectedTool === 'all'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Fusion Output</h2>
              <p class="section-subtitle">Fusion is shown only in All tools mode, as requested.</p>
            </div>
            <button class="btn btn-secondary" @click="loadFusion">Refresh Fusion</button>
          </div>

          <div v-if="fusionRows.length" class="table-scroll fusion-table">
            <table>
              <thead>
                <tr>
                  <th>Word</th>
                  <th>Lemma</th>
                  <th>Root</th>
                  <th>POS</th>
                  <th>Segmentation</th>
                  <th>Confidence</th>
                  <th>Sources</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in fusionRows" :key="`${row.word}-${row.index}`">
                  <td class="arabic" dir="rtl" lang="ar">{{ row.word }}</td>
                  <td>{{ value(row.final.lemma) }}</td>
                  <td>{{ value(row.final.root) }}</td>
                  <td>{{ value(row.final.pos) }}</td>
                  <td>{{ row.final.segmentation?.join(' + ') || '—' }}</td>
                  <td><span :class="['pill', confidencePill(row.final.confidence_level)]">{{ row.final.confidence_level || '—' }}</span></td>
                  <td>
                    <div class="chip-row">
                      <span
                        v-for="chip in fusionSourceChips(row.sources)"
                        :key="`${row.index}-${chip.label}`"
                        class="source-chip"
                        :style="{ backgroundColor: chip.color }"
                      >
                        {{ chip.label }}
                      </span>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="empty-state">Fusion output is not available for this run.</div>
        </div>

        <div v-if="activeTab === 'json'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Formatted JSON</h2>
              <p class="section-subtitle">Raw response for reproducibility and debugging.</p>
            </div>
          </div>
          <pre class="json-panel"><code>{{ prettyJson }}</code></pre>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import { API_BASE_URL, exportUrl } from '../api/nlpApi'
import { TOOL_CONFIG, TOOL_KEYS, toolOrder } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const rawResults = ref(null)
const fusionPayload = ref(null)
const selectedTool = ref('all')
const activeTab = ref('results')
const copied = ref(false)
const pendingTool = ref('')
const selectionNotice = ref(null)

const {
  toolStatuses,
  loading: statusLoading,
  error: statusError,
  refresh,
  toolStatus,
  toolReason,
} = useToolStatus()

const toolOptions = computed(() =>
  toolOrder(TOOL_KEYS).map((key) => ({
    key,
    ...TOOL_CONFIG[key],
  })),
)
const toolStatusesLoaded = computed(() => Object.keys(toolStatuses.value).length > 0)
const tokenEstimate = computed(() => (inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0))
const selectedToolMeta = computed(() => TOOL_CONFIG[selectedTool.value] || { label: 'All tools', color: '#0f172a', type: 'Combined analysis' })
const hasResults = computed(() => Boolean(rawResults.value || fusionPayload.value))
const jsonExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'json') : '#'))
const csvExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'csv') : '#'))
const prettyJson = computed(() => JSON.stringify({ analysis: rawResults.value, fusion: fusionPayload.value }, null, 2))
const tabs = computed(() => [
  { key: 'results', label: selectedTool.value === 'all' ? 'All tools' : 'Token breakdown' },
  { key: 'fusion', label: 'Fusion' },
  { key: 'json', label: 'JSON' },
])

function selectTool(toolKey) {
  selectionNotice.value = null
  pendingTool.value = ''

  if (toolKey === 'all') {
    selectedTool.value = 'all'
    return
  }

  const status = toolStatus(toolKey)
  if (status === 'ok' || status === 'lazy' || status === 'unknown') {
    selectedTool.value = toolKey
    selectionNotice.value =
      status === 'lazy'
        ? {
            kind: 'info',
            title: `${TOOL_CONFIG[toolKey].label} loads on demand`,
            message: 'This tool may be slower the first time it is selected.',
            pending: false,
          }
        : null
    return
  }

  pendingTool.value = toolKey
  selectionNotice.value = {
    kind: 'warn',
    title: `${TOOL_CONFIG[toolKey].label} is currently ${statusLabel(status)}`,
    message: toolReason(toolKey) || 'You can still choose it, but the backend is reporting a non-active state.',
    pending: true,
  }
}

function confirmPendingSelection() {
  if (!pendingTool.value) return
  selectedTool.value = pendingTool.value
  selectionNotice.value = null
  pendingTool.value = ''
}

function cancelPendingSelection() {
  pendingTool.value = ''
  selectionNotice.value = null
}

async function analyze() {
  if (!inputText.value.trim()) return
  if (statusLoading.value && !toolStatusesLoaded.value) await refresh()

  loading.value = true
  error.value = ''
  rawResults.value = null
  fusionPayload.value = null
  copied.value = false

  try {
    const response =
      selectedTool.value === 'all'
        ? await axios.get(`${API_BASE_URL}/analyze-combined`, { params: { text: inputText.value } })
        : await axios.get(`${API_BASE_URL}/analyze/${selectedTool.value}`, { params: { text: inputText.value } })

    rawResults.value = response.data
    if (selectedTool.value === 'all') {
      await loadFusion()
    }
  } catch (e) {
    error.value = readError(e, 'Failed to connect to the backend.')
  } finally {
    loading.value = false
  }
}

async function loadFusion() {
  if (!inputText.value.trim()) return
  try {
    const { data } = await axios.get(`${API_BASE_URL}/fusion`, { params: { text: inputText.value } })
    fusionPayload.value = normalizeFusionRows(data)
  } catch {
    fusionPayload.value = []
  }
}

function toolRows(toolKey) {
  const payload = selectedTool.value === 'all' ? rawResults.value?.[toolKey] : rawResults.value
  const tokens = Array.isArray(payload?.tokens) ? payload.tokens : []
  return tokens.map((token, index) => {
    const best = unwrapAnalysis(token)
    const values = {}
    for (const field of TOOL_CONFIG[toolKey].provides) {
      values[field] = readField(best, field)
    }
    return {
      index,
      surface: token.surface || token.word || `#${index + 1}`,
      values,
      confidence: confidenceFromToken(best),
    }
  })
}

function toolColumns(toolKey) {
  return TOOL_CONFIG[toolKey].provides
}

function normalizeFusionRows(payload) {
  const rows = Array.isArray(payload?.fusion_result?.fusion)
    ? payload.fusion_result.fusion
    : Array.isArray(payload?.fusion)
      ? payload.fusion
      : Array.isArray(payload?.result)
        ? payload.result
        : []

  return rows.map((row, index) => ({
    index,
    word: row?.word || row?.surface || `#${index + 1}`,
    final: row?.final || {},
    sources: row?.sources || {},
  }))
}

function fusionSourceChips(sources) {
  return Object.entries(sources || {}).map(([feature, tool]) => ({
    label: `${feature}: ${TOOL_CONFIG[tool]?.label || tool}`,
    color: TOOL_CONFIG[tool]?.color || '#64748b',
  }))
}

function unwrapAnalysis(token) {
  if (!token) return {}
  if (Array.isArray(token.analyses)) return token.analyses[0] || {}
  if (token.final) return token.final
  return token
}

function confidenceFromToken(token) {
  const raw = token?.confidence_level || token?.confidence
  if (raw) return String(raw).toLowerCase()
  const score = token?.confidence_score
  if (typeof score === 'number') {
    if (score >= 0.75) return 'high'
    if (score >= 0.45) return 'medium'
    return 'low'
  }
  return ''
}

function readField(raw, field) {
  if (!raw) return ''
  if (field === 'pos') return raw.pos || raw.upos || ''
  if (field === 'segmentation') return Array.isArray(raw.segmentation) ? raw.segmentation.join(' + ') : raw.segmentation || ''
  if (field === 'dependency') {
    const dependency = raw.dependency || raw.dep || {}
    const deprel = dependency.deprel || raw.deprel || ''
    return deprel ? (dependency.head_text ? `${deprel} -> ${dependency.head_text}` : deprel) : ''
  }
  return raw[field] || ''
}

function fieldLabel(field) {
  if (field === 'pos') return 'POS'
  if (field === 'segmentation') return 'Segmentation'
  if (field === 'dependency') return 'Dependency'
  if (field === 'case') return 'Case'
  if (field === 'definite') return 'Definite'
  if (field === 'gender') return 'Gender'
  if (field === 'number') return 'Number'
  if (field === 'tense') return 'Tense'
  if (field === 'gloss') return 'Gloss'
  if (field === 'stem') return 'Stem'
  if (field === 'lemma') return 'Lemma'
  if (field === 'root') return 'Root'
  return field
}

function isArabicField(field) {
  return ['lemma', 'root', 'gloss', 'stem'].includes(field)
}

function formatCellValue(field, value) {
  if (Array.isArray(value)) return value.join(' + ')
  if (value === null || value === undefined || value === '') return '—'
  return String(value)
}

function cellClass(field, value) {
  return value ? (isArabicField(field) ? 'field-value arabic' : 'field-value') : 'field-value muted'
}

function statusBadge(toolKey) {
  const status = toolStatus(toolKey)
  if (status === 'ok') return { show: false, label: '', className: '' }
  if (status === 'error') return { show: true, label: 'error', className: 'pill-red' }
  if (status === 'unavailable') return { show: true, label: 'unavailable', className: 'pill-gray' }
  if (status === 'lazy') return { show: true, label: '~700MB', className: 'pill-amber' }
  if (status === 'future_work') return { show: true, label: 'planned', className: 'pill-gray' }
  return { show: true, label: 'status unknown', className: 'pill-gray' }
}

function statusLabel(status) {
  if (status === 'ok') return 'active'
  if (status === 'error') return 'error'
  if (status === 'unavailable') return 'unavailable'
  if (status === 'lazy') return 'loads on demand'
  if (status === 'future_work') return 'planned'
  return 'status unknown'
}

function statusPill(status) {
  if (status === 'ok') return 'pill-green'
  if (status === 'error') return 'pill-red'
  if (status === 'unavailable' || status === 'future_work' || status === 'unknown') return 'pill-gray'
  return 'pill-amber'
}

function confidencePill(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'high') return 'pill-green'
  if (normalized === 'medium') return 'pill-blue'
  if (normalized === 'low') return 'pill-red'
  return 'pill-gray'
}

function value(item) {
  return item || '—'
}

function metricPercent(value) {
  const parsed = Number.parseFloat(String(value || '').replace('%', ''))
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0
}

function formatDecimal(value) {
  return typeof value === 'number' ? value.toFixed(3) : '0.000'
}

function readError(errorObject, fallback) {
  return errorObject?.response?.data?.detail || errorObject?.response?.data?.error || errorObject?.message || fallback
}

function loadSample() {
  inputText.value = 'وجدت المعلمة طالبة مجتهدة في الفصل'
}

function clear() {
  inputText.value = ''
  rawResults.value = null
  fusionPayload.value = null
  error.value = ''
  copied.value = false
  selectionNotice.value = null
  pendingTool.value = ''
  selectedTool.value = 'all'
  activeTab.value = 'results'
}

function guardExport(event) {
  if (!hasResults.value) event.preventDefault()
}

async function copyCurrentJson() {
  if (!hasResults.value) return
  await navigator.clipboard.writeText(prettyJson.value)
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}

onMounted(async () => {
  if (route.query.tool && TOOL_CONFIG[String(route.query.tool)]) {
    selectedTool.value = String(route.query.tool)
  }
  if (route.query.text) {
    inputText.value = String(route.query.text)
  }
  await nextTick()
  if (route.query.text) {
    analyze()
  }
})
</script>

<style scoped>
.analyze-hero {
  min-height: 250px;
}

.compact-actions,
.run-row {
  margin-top: 0;
}

.run-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
}

.disabled {
  pointer-events: none;
  opacity: 0.5;
}

.copy-note {
  color: var(--green);
  font-size: 13px;
  font-weight: 850;
}

.selector-panel {
  margin-top: 18px;
}

.selector-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.selector-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfdff;
  text-align: left;
  cursor: pointer;
  transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}

.selector-card:hover,
.selector-card.active {
  transform: translateY(-2px);
  border-color: rgba(37, 99, 235, 0.38);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}

.selector-card.all-tools {
  grid-column: 1 / -1;
  background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
}

.selector-dot {
  width: 12px;
  height: 12px;
  margin-top: 4px;
  border-radius: 999px;
  flex: 0 0 auto;
}

.selector-copy {
  display: grid;
  gap: 4px;
}

.selector-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.selector-label {
  color: var(--navy);
  font-size: 15px;
  font-weight: 950;
}

.selector-subtitle {
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}

.selection-notice {
  display: grid;
  gap: 8px;
  margin-top: 14px;
  padding: 14px;
  border-radius: 10px;
}

.selection-notice.info {
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  color: #1d4ed8;
}

.selection-notice.warn {
  border: 1px solid #fcd34d;
  background: #fffbeb;
  color: #92400e;
}

.analysis-loading {
  min-height: 170px;
}

.loading-stack {
  width: min(520px, 100%);
  display: grid;
  gap: 12px;
}

.loading-stack .wide {
  width: 78%;
}

.loading-stack .short {
  width: 42%;
}

.analysis-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 20px;
  padding: 5px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #eef3f8;
}

.analysis-tabs button {
  min-height: 38px;
  padding: 8px 14px;
  border-radius: 7px;
  color: var(--muted);
  background: transparent;
  cursor: pointer;
  font-weight: 900;
}

.analysis-tabs button.active {
  color: var(--navy);
  background: #fff;
  box-shadow: 0 1px 7px rgba(23, 32, 51, 0.09);
}

.tab-panel {
  display: grid;
  gap: 18px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.tool-result-card,
.single-result-card {
  display: grid;
  gap: 14px;
}

.tool-result-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-top: 10px;
  border-top: 4px solid transparent;
}

.tool-result-title {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.tool-result-bar {
  width: 12px;
  height: 42px;
  border-radius: 999px;
  flex: 0 0 auto;
}

.tool-result-title h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 950;
}

.tool-result-title p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 13px;
  font-weight: 750;
}

.tool-table {
  max-height: 420px;
}

.status-card {
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #fafbfc;
}

.status-card strong {
  display: block;
  font-weight: 950;
}

.status-card p {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}

.fusion-table {
  max-height: 520px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.source-chip {
  padding: 6px 10px;
  border-radius: 999px;
  color: #fff;
  font-size: 12px;
  font-weight: 900;
}

.field-value {
  color: var(--ink);
  font-size: 14px;
  font-weight: 850;
}

.field-value.arabic {
  font-size: 16px;
}

.field-value.muted {
  color: var(--muted);
}

.evaluation-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.metric-block {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.metric-block-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.metric-block strong {
  color: var(--navy);
  font-size: 20px;
  font-weight: 950;
}

.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metrics-note {
  margin: 14px 0 0;
  color: var(--muted);
}

.metric-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 60px;
  min-height: 34px;
  padding: 6px 10px;
  border-radius: 8px;
  background: #eef6ff;
  color: var(--navy);
  font-size: 15px;
  font-weight: 950;
}

.tool-chip-active {
  background: #15803d;
}

.tool-chip-muted {
  background: #94a3b8;
}

@media (max-width: 1100px) {
  .selector-grid,
  .results-grid,
  .evaluation-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .selector-grid,
  .results-grid,
  .evaluation-grid {
    grid-template-columns: 1fr;
  }

  .metric-main {
    align-items: flex-start;
  }
}
</style>
