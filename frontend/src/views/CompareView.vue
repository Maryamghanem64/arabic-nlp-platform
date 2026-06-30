<template>
  <div class="page-wrap compare-page page-stack">
    <section class="hero-band compare-hero">
      <div class="hero-content">
        <span class="eyebrow">Comparative NLP dashboard</span>
        <h1 class="hero-title">Unified token table from active analyzer outputs.</h1>
        <p class="hero-copy">
          This page shows raw analyzer outputs only. Each token occupies one row and each active analyzer occupies one dynamic column.
        </p>
        <p class="page-note">
          Results shown on this page are computed directly from analyzer outputs. No AI-generated interpretation is used.
        </p>
      </div>
    </section>

    <section class="panel panel-pad input-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
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
        placeholder="Enter Arabic text here..."
      ></textarea>

      <div class="run-row">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="compare">
          {{ loading ? 'Running comparison...' : 'Run comparison' }}
        </button>
        <button class="btn btn-secondary" :disabled="!hasResults" @click="copyResults">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

    <section v-if="statusError && !toolStatusesLoaded" class="error-state">
      <div>
        <strong>Backend status unavailable</strong>
        <p>{{ statusError.message || 'The comparison page could not load tool availability from GET /.' }}</p>
      </div>
    </section>

    <section v-if="loading" class="loading-grid">
      <div v-for="n in 4" :key="n" class="panel panel-pad skeleton-metric">
        <span class="skeleton"></span>
        <span class="skeleton wide"></span>
      </div>
    </section>

    <section v-if="error" class="error-state">
      <div>
        <strong>Comparison failed</strong>
        <p>{{ error }}</p>
        <button class="btn btn-secondary" @click="compare">Retry</button>
      </div>
    </section>

    <template v-if="hasResults && !loading">
      <section class="panel panel-pad section-block">
        <div class="section-head titled">
          <h2 class="section-title">Unified comparison table</h2>
          <span class="pill pill-blue">{{ activeColumnCount }} active analyzer{{ activeColumnCount === 1 ? '' : 's' }}</span>
        </div>

        <div v-if="comparisonRows.length && compareCols.length" class="table-scroll compare-scroll">
          <table class="comparison-table">
            <thead>
              <tr>
                <th class="sticky-word-col">Token</th>
                <th v-for="toolKey in compareCols" :key="toolKey" class="tool-col" :data-tool="toolKey">
                  <span class="header-tool">
                    <span class="header-dot" :style="{ backgroundColor: TOOL_CONFIG[toolKey].color }"></span>
                    <span>{{ TOOL_CONFIG[toolKey].label }}</span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in comparisonRows" :key="row.index">
                <td class="sticky-word-col word-cell">
                  <strong class="arabic" dir="rtl" lang="ar">{{ row.word }}</strong>
                  <span class="mono">#{{ row.index + 1 }}</span>
                </td>
                <td v-for="toolKey in compareCols" :key="`${row.index}-${toolKey}`" class="tool-col">
                  <div class="cell-stack">
                    <template v-if="row.tools[toolKey].available">
                      <span v-if="row.tools[toolKey].segmentation" class="cell-line mono">{{ row.tools[toolKey].segmentation }}</span>
                      <span v-if="row.tools[toolKey].lemma" class="cell-line arabic" dir="rtl" lang="ar">{{ row.tools[toolKey].lemma }}</span>
                      <span v-if="row.tools[toolKey].pos" class="cell-line">{{ row.tools[toolKey].pos }}</span>
                      <span v-if="row.tools[toolKey].root" class="cell-line arabic" dir="rtl" lang="ar">{{ row.tools[toolKey].root }}</span>
                      <span v-if="row.tools[toolKey].confidence" class="cell-line">Confidence: {{ row.tools[toolKey].confidence }}</span>
                      <span v-if="row.tools[toolKey].processingTime" class="cell-line">Processing: {{ row.tools[toolKey].processingTime }}</span>
                    </template>
                    <EmptyCell v-else />
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty-state">No comparison rows were returned, or no compare-enabled tools are currently online.</div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import EmptyCell from '../components/tables/EmptyCell.vue'
import { API_BASE_URL, exportUrl } from '../api/nlpApi'
import { TOOL_CONFIG, toolOrder } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'
import { canonicalToken } from '../utils/tokenModel'
import { recordAnalysis } from '../utils/analysisHistory'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const comparisonPayload = ref(null)
const copied = ref(false)

const { toolStatuses, activeTools, loading: statusLoading, error: statusError, refresh } = useToolStatus()

const compareEnabledTools = ['camel', 'farasa', 'stanza', 'qalsadi', 'alkhalil', 'udpipe']
const activeColumnKeys = computed(() => {
  const fromPayload = Object.keys(comparisonPayload.value?.tools || {})
  const fromRows = (comparisonPayload.value?.comparison || []).flatMap((row) => Object.keys(row?.tools || {}))
  return [...new Set([...fromPayload, ...fromRows])].filter((key) => compareEnabledTools.includes(key))
})
const compareCols = computed(() => {
  const available = toolOrder(activeTools.value).filter((key) => compareEnabledTools.includes(key))
  return activeColumnKeys.value.length ? activeColumnKeys.value : available
})
const tokenEstimate = computed(() => (inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0))
const comparisonRows = computed(() => normalizeComparisonRows(comparisonPayload.value, compareCols.value))
const hasResults = computed(() => Boolean(comparisonRows.value.length))
const toolStatusesLoaded = computed(() => Object.keys(toolStatuses.value).length > 0)
const activeColumnCount = computed(() => compareCols.value.length)
const jsonExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'json') : '#'))
const csvExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'csv') : '#'))

async function compare() {
  if (!inputText.value.trim()) return
  if (statusLoading.value && !toolStatusesLoaded.value) await refresh()
  if (!compareCols.value.length) {
    error.value = 'No compare-enabled tools are currently online.'
    return
  }

  loading.value = true
  error.value = ''
  comparisonPayload.value = null
  copied.value = false

  try {
    const { data } = await axios.get(`${API_BASE_URL}/compare`, {
      params: { text: inputText.value, tools: compareCols.value.join(',') },
    })
    comparisonPayload.value = data
    recordAnalysis({
      page: 'Compare',
      text: inputText.value.trim(),
      summary: `${compareCols.value.length} tools | ${(data?.comparison || []).length} tokens`,
    })
  } catch (e) {
    error.value = e?.response?.data?.detail || e?.response?.data?.error || e?.message || 'Failed to connect to the backend.'
  } finally {
    loading.value = false
  }
}

async function copyResults() {
  if (!hasResults.value) return
  await navigator.clipboard.writeText(JSON.stringify(comparisonPayload.value, null, 2))
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}

function guardExport(event) {
  if (!hasResults.value) event.preventDefault()
}

function loadSample() {
  inputText.value = '\u0642\u0631\u0623 \u0627\u0644\u0637\u0627\u0644\u0628 \u0627\u0644\u0643\u062a\u0628 \u0641\u064a \u0627\u0644\u0645\u0643\u062a\u0628\u0629'
}

function clear() {
  inputText.value = ''
  comparisonPayload.value = null
  error.value = ''
  copied.value = false
}

function normalizeComparisonRows(payload, cols) {
  const rows = Array.isArray(payload?.comparison) ? payload.comparison : []
  return rows.map((row, index) => {
    const tools = row?.tools || {}
    const normalizedTools = Object.fromEntries(
      cols.map((toolKey) => [toolKey, normalizeToolCell(toolKey, tools[toolKey], payload?.tools?.[toolKey])]),
    )
    return {
      index,
      word: row?.word || row?.surface || `#${index + 1}`,
      tools: normalizedTools,
    }
  })
}

function normalizeToolCell(toolKey, rawToken, toolPayload) {
  const best = canonicalToken(rawToken)
  const available = Boolean(
    best?.surface ||
      best?.lemma ||
      best?.root ||
      best?.pos ||
      best?.segmentation ||
      best?.confidence?.level ||
      best?.confidence?.score,
  )

  return {
    available,
    segmentation: formatSegmentation(best),
    lemma: best.lemma || '',
    pos: best.pos || best.upos || '',
    root: best.root || '',
    confidence: confidenceValue(best),
    processingTime: processingTimeValue(toolKey, toolPayload),
  }
}

function confidenceValue(token) {
  const level = token?.confidence?.level
  const score = token?.confidence?.score
  if (level) return String(level)
  if (typeof score === 'number') return `${Math.round(score * 100)}%`
  return ''
}

function processingTimeValue(toolKey, toolPayload) {
  const payload = toolPayload || comparisonPayload.value?.tools?.[toolKey] || {}
  const runtime = payload.runtime_ms ?? payload.elapsed ?? payload.elapsed_ms ?? payload.processing_time_ms ?? payload.processing_time
  if (runtime === null || runtime === undefined || runtime === '') return ''
  if (typeof runtime === 'number') return `${Math.round(runtime)} ms`
  const parsed = Number.parseFloat(String(runtime))
  return Number.isFinite(parsed) ? `${Math.round(parsed)} ms` : String(runtime)
}

function formatSegmentation(token) {
  if (Array.isArray(token?.segmentation) && token.segmentation.length) return token.segmentation.join(' + ')
  if (Array.isArray(token?.segments) && token.segments.length) return token.segments.join(' + ')
  if (Array.isArray(token?.parts) && token.parts.length) return token.parts.join(' + ')
  return token?.segmentation || token?.segments || token?.parts || ''
}

onMounted(() => {
  if (route.query.text) {
    inputText.value = String(route.query.text)
    compare()
  }
})
</script>

<style scoped>
.compare-hero {
  min-height: 250px;
}

.compact-actions {
  margin-top: 0;
}

.run-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
}

.copy-note {
  color: var(--green);
  font-size: 13px;
  font-weight: 500;
}

.disabled {
  pointer-events: none;
  opacity: 0.5;
}

.page-note {
  margin: 10px 0 0;
  padding: 10px 12px;
  border-left: 3px solid var(--c-accent-border);
  border-radius: 10px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 13px;
  line-height: 1.5;
}

.compare-scroll {
  overflow-x: auto;
}

.comparison-table {
  width: max-content;
  min-width: 100%;
  table-layout: fixed;
}

.comparison-table th,
.comparison-table td {
  width: 180px;
}

.comparison-table .sticky-word-col {
  position: sticky;
  left: 0;
  z-index: 2;
  width: 130px;
  min-width: 130px;
  max-width: 130px;
  background: var(--c-surface);
}

.comparison-table thead .sticky-word-col {
  z-index: 4;
}

.comparison-table thead th {
  position: sticky;
  top: 0;
  z-index: 3;
  background: var(--c-page-bg);
}

.header-tool {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.header-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  flex: 0 0 auto;
}

.word-cell {
  display: grid;
  gap: 5px;
}

.word-cell strong {
  font-size: 18px;
  font-weight: 500;
}

.word-cell .mono {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}

.tool-col {
  min-width: 180px;
  max-width: 180px;
  vertical-align: top;
}

.cell-stack {
  display: grid;
  gap: 6px;
}

.cell-line {
  display: block;
  overflow: hidden;
  color: var(--ink);
  font-size: 13px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cell-line.arabic {
  font-size: 14px;
}

.empty-state {
  padding: 16px;
  border: 1px dashed var(--c-border-strong);
  border-radius: var(--radius-card);
  color: var(--muted);
  background: var(--c-page-bg);
}

.section-block {
  scroll-margin-top: 92px;
}

.loading-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.skeleton-metric {
  display: grid;
  gap: 12px;
}

.skeleton-metric .wide {
  width: 68%;
}

@media (max-width: 1100px) {
  .loading-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .comparison-table th,
  .comparison-table td {
    width: 170px;
  }
}

@media (max-width: 720px) {
  .loading-grid {
    grid-template-columns: 1fr;
  }
}
</style>
