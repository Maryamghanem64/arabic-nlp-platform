<template>
  <div class="page-wrap compare-page page-stack">
    <section class="hero-band compare-hero">
      <div class="hero-content">
        <span class="eyebrow">Tool output comparison lab</span>
        <h1 class="hero-title">Inspect aligned analyzer evidence and isolate disagreements.</h1>
        <p class="hero-copy">
          Each aligned token is inspected feature by feature. Agreement and conflict summaries include only analyzers eligible for that linguistic capability; the raw table remains unchanged and no gold-standard correctness is claimed.
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
      <section class="comparison-summary-grid" aria-label="Comparison summary">
        <article class="summary-card"><span>Aligned tokens</span><strong>{{ comparisonRows.length }}</strong></article>
        <article class="summary-card"><span>Displayed analyzers</span><strong>{{ activeColumnCount }}</strong></article>
        <article class="summary-card"><span>Capability-eligible conflicts</span><strong :class="{ 'text-conflict': conflictCount }">{{ conflictCount }}</strong></article>
        <article class="summary-card"><span>Comparable token-features</span><strong>{{ comparablePairCount }}</strong></article>
      </section>

      <section class="panel panel-pad section-block">
        <div class="view-mode-row">
          <div>
            <span class="eyebrow">Research view</span>
            <h2 class="section-title">Capability-scoped comparison evidence</h2>
          </div>
          <div class="view-switch" role="tablist" aria-label="Comparison view">
            <button v-for="mode in viewModes" :key="mode.key" type="button" :class="{ active: activeView === mode.key }" @click="activeView = mode.key">{{ mode.label }}</button>
          </div>
        </div>

        <div class="scope-note">
          <strong>Scope:</strong>
          the aligned table displays the fixed comparison schema including SinaTools. If SinaTools is not aligned by the backend but is returned by <code>/analyze-combined</code>, its cell is filled from the actual combined analyzer output and marked as such.
        </div>

        <template v-if="activeView === 'aligned'">
        <div class="section-head titled">
          <h2 class="section-title">Aligned analyzer evidence</h2>
          <span class="pill pill-blue">{{ activeReturnedCount }} returned analyzer{{ activeReturnedCount === 1 ? '' : 's' }} · {{ compareCols.length }} columns shown</span>
        </div>

        <p class="aligned-note">
          The table shows aligned evidence plus eligible fallback evidence from combined analysis when a tool is not present in the alignment payload.
          Empty cells mean the tool did not return usable evidence for that token.
        </p>

        <div v-if="comparisonRows.length && compareCols.length" class="table-scroll compare-scroll">
          <table class="comparison-table">
            <thead>
              <tr>
                <th class="sticky-word-col">Token</th>
                <th v-for="toolKey in compareCols" :key="toolKey" class="tool-col" :data-tool="toolKey">
                  <span class="header-tool">
                    <span class="header-dot" :style="{ backgroundColor: toolMeta(toolKey).color }"></span>
                    <span>{{ toolMeta(toolKey).label }}</span>
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
                <td v-for="toolKey in compareCols" :key="`${row.index}-${toolKey}`" :class="['tool-col', cellAgreementClass(row, toolKey)]">
                  <div class="cell-stack">
                    <template v-if="row.tools[toolKey].available">
                      <div
                        v-for="item in compareCellItems(toolKey, row.tools[toolKey])"
                        :key="`${row.index}-${toolKey}-${item.feature}`"
                        class="cell-evidence-row"
                      >
                        <span class="cell-evidence-label">{{ item.label }}</span>
                        <span
                          :class="featureClass(item.feature)"
                          :dir="featureDirection(item.feature)"
                          :lang="featureDirection(item.feature) === 'rtl' ? 'ar' : null"
                        >
                          {{ item.value }}
                        </span>
                      </div>
                      <span v-if="row.tools[toolKey].source === 'combined'" class="source-note">from combined output</span>
                    </template>
                    <EmptyCell v-else label="Not aligned / not returned" />
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty-state">No comparison rows were returned, or no compare-enabled tools are currently online.</div>
        </template>

        <section v-else-if="activeView === 'conflicts'" class="conflict-list">
          <article v-for="conflict in conflictRows" :key="conflict.id" class="conflict-card">
            <div class="conflict-card-head">
              <div><span class="mono">#{{ conflict.index + 1 }}</span><strong class="arabic" dir="rtl" lang="ar">{{ conflict.word }}</strong></div>
              <span class="conflict-type-pill">{{ conflictTitle(conflict.feature) }}</span>
            </div>
            <div class="conflict-values-grid">
              <div v-for="item in conflict.values" :key="item.tool" class="conflict-value">
                <span class="header-tool"><span class="header-dot" :style="{ backgroundColor: toolMeta(item.tool).color }"></span>{{ toolMeta(item.tool).label }}</span>
                <strong :class="{ arabic: ['lemma', 'root', 'segmentation'].includes(conflict.feature) }">{{ item.value }}</strong>
              </div>
            </div>
            <p>{{ conflictExplanation(conflict.feature) }}</p>
          </article>
          <div v-if="!conflictRows.length" class="empty-state">No feature-level disagreements were detected in comparable analyzer outputs.</div>
        </section>

        <section v-else-if="activeView === 'summary'" class="agreement-summary">
          <article v-for="metric in agreementMetrics" :key="metric.feature" class="agreement-row">
            <div><strong>{{ metric.label }}</strong><span>{{ metric.comparable }} comparable token{{ metric.comparable === 1 ? '' : 's' }}</span></div>
            <div class="agreement-meter"><span :style="{ width: `${metric.percent}%` }"></span></div>
            <strong>{{ metric.percent }}%</strong>
          </article>
        </section>

        <pre v-else class="raw-json">{{ prettyComparisonJson }}</pre>
      </section>

      <section class="mobile-comparison-list" aria-label="Mobile token comparison">
        <article v-for="row in comparisonRows" :key="`mobile-${row.index}`" class="mobile-token-card">
          <header><span class="mono">#{{ row.index + 1 }}</span><strong class="arabic" dir="rtl" lang="ar">{{ row.word }}</strong></header>
          <div v-for="toolKey in compareCols" :key="`mobile-${row.index}-${toolKey}`" class="mobile-tool-row">
            <span class="header-tool"><span class="header-dot" :style="{ backgroundColor: toolMeta(toolKey).color }"></span>{{ toolMeta(toolKey).label }}</span>
            <div v-if="row.tools[toolKey].available" class="mobile-evidence">
              <span
                v-for="item in compareCellItems(toolKey, row.tools[toolKey])"
                :key="`mobile-${row.index}-${toolKey}-${item.feature}`"
              >
                {{ item.label }}:
                <b
                  :class="{ arabic: ['lemma', 'root', 'segmentation'].includes(item.feature) }"
                  :dir="['lemma', 'root', 'segmentation'].includes(item.feature) ? 'rtl' : null"
                >
                  {{ item.value }}
                </b>
              </span>
              <span v-if="row.tools[toolKey].source === 'combined'" class="source-note">from combined output</span>
            </div>
            <EmptyCell v-else label="Not aligned / not returned" />
          </div>
          <span v-if="rowConflictCount(row)" class="mobile-conflict-note">{{ rowConflictCount(row) }} feature conflict{{ rowConflictCount(row) === 1 ? '' : 's' }}</span>
        </article>
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
import { TOOL_CONFIG, FEATURE_ELIGIBILITY, toolOrder } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'
import { canonicalToken } from '../utils/tokenModel'
import { recordAnalysis } from '../utils/analysisHistory'

function toolMeta(toolKey) {
  return TOOL_CONFIG?.[toolKey] || {
    label: toolKey || 'Unknown tool',
    color: '#64748B',
    type: 'Optional analyzer',
    provides: [],
  }
}

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const comparisonPayload = ref(null)
const combinedPayload = ref(null)
const copied = ref(false)
const activeView = ref('aligned')
const viewModes = [
  { key: 'aligned', label: 'Aligned table' },
  { key: 'conflicts', label: 'Conflict view' },
  { key: 'summary', label: 'Eligible agreement summary' },
  { key: 'raw', label: 'Raw JSON' },
]

const { toolStatuses, activeTools, loading: statusLoading, error: statusError, refresh } = useToolStatus()

const compareEnabledTools = ['camel', 'alkhalil', 'sinatools', 'stanza', 'udpipe', 'farasa', 'qalsadi']

const FEATURE_ELIGIBILITY_FALLBACK = {
  lemma: ['camel', 'alkhalil', 'sinatools', 'stanza', 'udpipe', 'qalsadi'],
  root: ['camel', 'alkhalil', 'sinatools'],
  pos: ['camel', 'alkhalil', 'sinatools', 'stanza', 'udpipe'],
}

const activeColumnKeys = computed(() => {
  const keys = new Set()

  Object.keys(comparisonPayload.value?.tools || {}).forEach((key) => keys.add(key))
  ;(comparisonPayload.value?.comparison || []).forEach((row) => {
    Object.keys(row?.tools || {}).forEach((key) => keys.add(key))
  })

  const combinedTools = combinedToolsPayload()
  Object.keys(combinedTools || {}).forEach((key) => {
    const tokens = combinedTools?.[key]?.tokens
    if (Array.isArray(tokens) && tokens.length) keys.add(key)
  })

  return toolOrder([...keys]).filter((key) => compareEnabledTools.includes(key))
})

const compareCols = computed(() =>
  toolOrder([...new Set([...compareEnabledTools, ...activeColumnKeys.value])])
    .filter((key) => compareEnabledTools.includes(key)),
)
const tokenEstimate = computed(() => (inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0))
const comparisonRows = computed(() => normalizeComparisonRows(comparisonPayload.value, compareCols.value))
const hasResults = computed(() => Boolean(comparisonRows.value.length))
const toolStatusesLoaded = computed(() => Object.keys(toolStatuses.value).length > 0)
const activeColumnCount = computed(() => compareCols.value.length)
const activeReturnedCount = computed(() => activeColumnKeys.value.length || activeTools.value.filter((key) => compareEnabledTools.includes(key)).length)
const jsonExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'json') : '#'))
const csvExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'csv') : '#'))
const comparableFeatures = ['lemma', 'pos', 'root']

function eligibleCompareTools(feature) {
  const eligible = new Set(FEATURE_ELIGIBILITY[feature] || [])
  return compareCols.value.filter((tool) => eligible.has(tool))
}
const featureComparisons = computed(() => {
  const result = Object.fromEntries(comparableFeatures.map((feature) => [feature, { comparable: 0, agree: 0, conflicts: [] }]))
  comparisonRows.value.forEach((row) => {
    comparableFeatures.forEach((feature) => {
      const values = eligibleCompareTools(feature)
        .map((tool) => ({ tool, value: normalizedComparableValue(feature, row.tools[tool]?.[feature]) }))
        .filter((item) => item.value)
      if (values.length < 2) return
      result[feature].comparable += 1
      const unique = [...new Set(values.map((item) => item.value))]
      if (unique.length === 1) result[feature].agree += 1
      else result[feature].conflicts.push({ row, values })
    })
  })
  return result
})
const conflictRows = computed(() => comparableFeatures.flatMap((feature) =>
  featureComparisons.value[feature].conflicts.map(({ row, values }, index) => ({
    id: `${row.index}-${feature}-${index}`, index: row.index, word: row.word, feature, values,
  })),
))
const conflictCount = computed(() => conflictRows.value.length)
const comparablePairCount = computed(() => comparableFeatures.reduce((sum, feature) => sum + featureComparisons.value[feature].comparable, 0))
const agreementMetrics = computed(() => comparableFeatures.map((feature) => {
  const metric = featureComparisons.value[feature]
  return {
    feature,
    label: feature === 'pos' ? 'POS agreement' : `${feature.charAt(0).toUpperCase()}${feature.slice(1)} agreement`,
    comparable: metric.comparable,
    percent: metric.comparable ? Math.round((metric.agree / metric.comparable) * 100) : 0,
  }
}))
const prettyComparisonJson = computed(() => JSON.stringify(comparisonPayload.value, null, 2))


async function compare() {
  if (!inputText.value.trim()) return
  if (statusLoading.value && !toolStatusesLoaded.value) await refresh()

  loading.value = true
  error.value = ''
  comparisonPayload.value = null
  combinedPayload.value = null
  copied.value = false

  try {
    const toolParam = compareEnabledTools.join(',')
    const [compareResponse, combinedResponse] = await Promise.all([
      axios.get(`${API_BASE_URL}/compare`, {
        params: { text: inputText.value, tools: toolParam },
      }),
      axios.get(`${API_BASE_URL}/analyze-combined`, {
        params: { text: inputText.value },
      }).catch(() => ({ data: null })),
    ])

    comparisonPayload.value = compareResponse.data
    combinedPayload.value = combinedResponse.data

    recordAnalysis({
      page: 'Compare',
      text: inputText.value.trim(),
      summary: `${compareCols.value.length} displayed analyzers | ${(compareResponse.data?.comparison || []).length} aligned tokens`,
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
  combinedPayload.value = null
  error.value = ''
  copied.value = false
}

function normalizeComparisonRows(payload, cols) {
  const rows = Array.isArray(payload?.comparison) ? payload.comparison : []
  return rows.map((row, index) => {
    const tools = row?.tools || {}
    const normalizedTools = Object.fromEntries(
      cols.map((toolKey) => {
        const rawAligned = tools[toolKey]
        const rawFallback = fallbackCombinedToken(toolKey, index)
        return [toolKey, normalizeToolCell(toolKey, rawAligned, payload?.tools?.[toolKey], rawFallback)]
      }),
    )
    return {
      index,
      word: row?.word || row?.surface || fallbackSurface(index) || `#${index + 1}`,
      tools: normalizedTools,
    }
  })
}

function combinedToolsPayload() {
  const payload = combinedPayload.value
  return payload?.tools || payload?.combined?.tools || payload?.combined || {}
}

function fallbackCombinedToken(toolKey, index) {
  const payload = combinedToolsPayload()?.[toolKey]
  const tokens = Array.isArray(payload?.tokens) ? payload.tokens : []
  return tokens[index] || null
}

function fallbackSurface(index) {
  const tools = combinedToolsPayload()
  for (const payload of Object.values(tools || {})) {
    const token = Array.isArray(payload?.tokens) ? payload.tokens[index] : null
    if (token?.surface || token?.word) return token.surface || token.word
  }
  return ''
}

function normalizeToolCell(toolKey, rawToken, toolPayload, fallbackToken = null) {
  const sourceToken = rawToken || fallbackToken
  const best = canonicalToken(sourceToken)
  const status = toolPayload?.status || combinedToolsPayload()?.[toolKey]?.status || ''
  const available = Boolean(
    best?.surface ||
      best?.lemma ||
      best?.root ||
      best?.pos ||
      best?.upos ||
      best?.segmentation ||
      best?.segments ||
      best?.parts ||
      best?.confidence?.level ||
      best?.confidence?.score,
  )

  return {
    available,
    source: rawToken ? 'aligned' : fallbackToken ? 'combined' : status || 'missing',
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

function compareCellItems(toolKey, cell) {
  if (!cell?.available) return []

  const order = {
    farasa: ['segmentation'],
    camel: ['lemma', 'root', 'pos', 'confidence'],
    sinatools: ['lemma', 'pos', 'root', 'confidence'],
    alkhalil: ['lemma', 'root', 'pos', 'confidence'],
    stanza: ['lemma', 'pos', 'confidence'],
    udpipe: ['lemma', 'pos', 'confidence'],
    qalsadi: ['lemma'],
  }[toolKey] || ['lemma', 'pos', 'root', 'segmentation']

  return order
    .map((feature) => ({
      feature,
      label: featureLabel(feature),
      value: cell?.[feature] || '',
    }))
    .filter((item) => item.value)
}

function featureLabel(feature) {
  const labels = {
    lemma: 'Lemma',
    root: 'Root',
    pos: 'POS',
    segmentation: 'Seg.',
    confidence: 'Conf.',
    processingTime: 'Time',
  }
  return labels[feature] || feature
}

function featureClass(feature) {
  if (['lemma', 'root', 'segmentation'].includes(feature)) return 'arabic cell-evidence-value'
  if (feature === 'pos') return 'pos-badge cell-pos-value'
  return 'cell-evidence-value'
}

function featureDirection(feature) {
  return ['lemma', 'root', 'segmentation'].includes(feature) ? 'rtl' : null
}

function normalizedComparableValue(feature, value) {
  if (value === null || value === undefined || value === '') return ''
  const raw = String(value).trim().toLowerCase()
  if (!raw) return ''
  if (feature === 'segmentation') return raw.replace(/\s+/g, '').replace(/\|/g, '+')
  if (feature === 'lemma' || feature === 'root') return raw.replace(/[ًٌٍَُِّْـ]/g, '').replace(/[أإآ]/g, 'ا').replace(/ى/g, 'ي')
  return raw
}

function rowConflictCount(row) {
  return comparableFeatures.reduce((count, feature) => {
    const values = eligibleCompareTools(feature).map((tool) => normalizedComparableValue(feature, row.tools[tool]?.[feature])).filter(Boolean)
    return count + (values.length >= 2 && new Set(values).size > 1 ? 1 : 0)
  }, 0)
}

function cellAgreementClass(row, toolKey) {
  const cell = row.tools[toolKey]
  if (!cell?.available) return 'cell-na'
  const hasConflict = comparableFeatures.some((feature) => {
    const own = normalizedComparableValue(feature, cell[feature])
    if (!own) return false
    const values = eligibleCompareTools(feature).map((tool) => normalizedComparableValue(feature, row.tools[tool]?.[feature])).filter(Boolean)
    return values.length >= 2 && new Set(values).size > 1
  })
  if (hasConflict) return 'cell-conflict'
  const hasComparableAgreement = comparableFeatures.some((feature) => {
    const own = normalizedComparableValue(feature, cell[feature])
    if (!own) return false
    const values = eligibleCompareTools(feature).map((tool) => normalizedComparableValue(feature, row.tools[tool]?.[feature])).filter(Boolean)
    return values.length >= 2 && new Set(values).size === 1
  })
  return hasComparableAgreement ? 'cell-agree' : ''
}

function conflictTitle(feature) {
  if (feature === 'lemma') return 'Lemma convention difference'
  if (feature === 'pos') return 'POS disagreement'
  if (feature === 'root') return 'Root disagreement'
  if (feature === 'segmentation') return 'Segmentation convention'
  return `${feature} disagreement`
}

function conflictExplanation(feature) {
  const explanations = {
    pos: 'Different POS conventions or contextual disambiguation strategies can produce different labels for the same Arabic surface form.',
    lemma: 'Lemma mismatches may reflect diacritization, normalization, or analyzer-specific lexical conventions.',
    root: 'Root extraction is not uniformly supported and analyzers may apply different morphological assumptions.',
    segmentation: 'Segmentation differences usually reflect clitic-boundary and tokenization conventions.',
  }
  return explanations[feature] || 'Analyzer outputs disagree for this comparable feature.'
}

function toolMetaLabel(toolKey) {
  return toolMeta(toolKey).label
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

.comparison-summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.summary-card { display: grid; gap: 6px; padding: 16px; border: 1px solid var(--c-border); border-radius: var(--radius-card); background: var(--c-surface); }
.summary-card span { color: var(--c-text-secondary); font-size: .8rem; }
.summary-card strong { font-size: 1.45rem; }
.text-conflict { color: #b91c1c; }
.view-mode-row { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 18px; }
.view-switch { display: flex; flex-wrap: wrap; gap: 6px; }
.view-switch button { border: 1px solid var(--c-border); background: var(--c-surface); color: var(--c-text-secondary); padding: 8px 11px; border-radius: 999px; cursor: pointer; }
.view-switch button.active { color: var(--c-accent); border-color: var(--c-accent); background: var(--c-accent-light); font-weight: 700; }
.tool-col.cell-agree { background: rgba(34, 197, 94, .055); }
.tool-col.cell-conflict { background: rgba(245, 158, 11, .09); box-shadow: inset 3px 0 0 #d97706; }
.tool-col.cell-na { background: rgba(148, 163, 184, .06); }
.conflict-list { display: grid; gap: 12px; }
.conflict-card { padding: 16px; border: 1px solid #fed7aa; border-left: 4px solid #d97706; border-radius: var(--radius-control); background: #fffbeb; }
.conflict-card-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.conflict-card-head > div { display: flex; gap: 10px; align-items: center; }
.conflict-values-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; margin: 14px 0 10px; }
.conflict-value { display: grid; gap: 6px; padding: 10px; border: 1px solid rgba(217,119,6,.22); border-radius: 10px; background: white; }
.conflict-card p { margin: 0; color: #78350f; line-height: 1.55; }
.agreement-summary { display: grid; gap: 12px; }
.agreement-row { display: grid; grid-template-columns: minmax(150px, 220px) 1fr 58px; gap: 14px; align-items: center; }
.agreement-row > div:first-child { display: grid; gap: 3px; }
.agreement-row span { color: var(--c-text-secondary); font-size: .78rem; }
.agreement-meter { height: 10px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
.agreement-meter span { display: block; height: 100%; background: var(--c-accent); border-radius: inherit; }
.raw-json { max-height: 620px; overflow: auto; margin: 0; padding: 16px; border-radius: var(--radius-control); background: #0f172a; color: #e2e8f0; font-size: .78rem; line-height: 1.55; }
.mobile-comparison-list { display: none; }

@media (max-width: 1100px) {
  .comparison-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .loading-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .comparison-table th,
  .comparison-table td {
    width: 170px;
  }
}

@media (max-width: 720px) {
  .comparison-summary-grid { grid-template-columns: 1fr 1fr; }
  .view-mode-row { align-items: flex-start; flex-direction: column; }
  .compare-scroll { display: none; }
  .mobile-comparison-list { display: grid; gap: 12px; }
  .mobile-token-card { padding: 14px; border: 1px solid var(--c-border); border-radius: var(--radius-card); background: var(--c-surface); }
  .mobile-token-card header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--c-border); }
  .mobile-token-card header strong { font-size: 1.15rem; }
  .mobile-tool-row { display: grid; gap: 7px; padding: 11px 0; border-bottom: 1px solid var(--c-border); }
  .mobile-evidence { display: flex; flex-wrap: wrap; gap: 6px 12px; color: var(--c-text-secondary); font-size: .82rem; }
  .mobile-conflict-note { display: inline-flex; margin-top: 10px; color: #92400e; font-size: .8rem; font-weight: 700; }
  .agreement-row { grid-template-columns: 1fr 52px; }
  .agreement-meter { grid-column: 1 / -1; grid-row: 2; }
  .conflict-card-head { align-items: flex-start; flex-direction: column; }
  .loading-grid {
    grid-template-columns: 1fr;
  }
}

.scope-note {
  margin-bottom: 14px;
  padding: 12px 14px;
  border-left: 3px solid var(--c-accent);
  background: var(--c-accent-light);
  color: var(--c-text-secondary);
  font-size: 13px;
  line-height: 1.6;
}
.scope-note strong { color: var(--c-accent-text); }

.cell-evidence-row {
  display: grid;
  grid-template-columns: 78px minmax(0, 1fr);
  gap: 8px;
  align-items: baseline;
}

.cell-evidence-label {
  color: var(--c-text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.cell-evidence-value {
  overflow-wrap: anywhere;
  color: var(--c-text-primary);
  font-size: 13px;
  font-weight: 600;
}

.cell-pos-value {
  width: fit-content;
  min-height: 22px;
  padding: 2px 7px;
}

.source-note {
  display: inline-flex;
  width: fit-content;
  padding: 3px 7px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .03em;
  text-transform: uppercase;
}

.scope-note code {
  color: var(--c-accent-text);
  font-weight: 700;
}

.comparison-table th,
.comparison-table td {
  width: 205px;
}

.tool-col {
  min-width: 205px;
  max-width: 205px;
}


/* FINAL COMPARE RESEARCH PASS */
.compare-page {
  width: min(96vw, 1500px);
}

.aligned-note {
  margin: 0 0 12px;
  padding: 10px 12px;
  border: 1px solid var(--c-border);
  border-left: 3px solid var(--c-accent);
  border-radius: 10px;
  background: var(--c-page-bg);
  color: var(--c-text-secondary);
  font-size: 12.5px;
  line-height: 1.55;
}

.compare-scroll {
  border: 1px solid var(--c-border);
  border-radius: 12px;
  background: var(--c-surface);
}

.comparison-table {
  width: max-content;
  min-width: 100%;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: 0;
}

.comparison-table th,
.comparison-table td {
  width: 235px;
  min-width: 235px;
  max-width: 235px;
  padding: 12px;
  border-right: 1px solid var(--c-border);
  border-bottom: 1px solid var(--c-border);
}

.comparison-table .sticky-word-col {
  width: 135px;
  min-width: 135px;
  max-width: 135px;
  background: #fff;
  box-shadow: 3px 0 8px rgba(15, 23, 42, .04);
}

.comparison-table thead th {
  background: #f8fafc;
  color: var(--c-text-secondary);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.header-tool {
  max-width: 100%;
  min-width: 0;
}

.header-tool span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tool-col {
  min-width: 235px;
  max-width: 235px;
}

.cell-stack {
  gap: 7px;
  min-width: 0;
}

.cell-evidence-row {
  display: grid;
  grid-template-columns: 54px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
  min-width: 0;
}

.cell-evidence-label {
  color: #6b84a4;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .055em;
  text-transform: uppercase;
}

.cell-evidence-value {
  min-width: 0;
  overflow: visible;
  overflow-wrap: anywhere;
  white-space: normal;
  color: var(--c-text-primary);
  font-size: 12.5px;
  line-height: 1.45;
  font-weight: 600;
}

.cell-evidence-value.arabic,
.arabic.cell-evidence-value {
  font-size: 14px;
  line-height: 1.65;
  text-align: right;
}

.cell-pos-value {
  display: inline-flex;
  width: fit-content;
  min-height: 20px;
  padding: 2px 7px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 10.5px;
  font-weight: 800;
}

.source-note {
  margin-top: 2px;
  background: #f1f5f9;
  color: #64748b;
  border: 1px solid #e2e8f0;
}

.tool-col.cell-agree {
  background: #f6faf8;
  box-shadow: inset 3px 0 0 #6f9a8d;
}

.tool-col.cell-conflict {
  background: #fffaf0;
  box-shadow: inset 3px 0 0 #c4924a;
}

.tool-col.cell-na {
  background: #f8fafc;
  color: var(--c-text-muted);
}

.conflict-list {
  gap: 10px;
}

.conflict-card {
  padding: 13px 14px;
  border: 1px solid #d9e2ec;
  border-left: 4px solid #8aa0b7;
  border-radius: 12px;
  background: #fff;
}

.conflict-card-head {
  align-items: flex-start;
}

.conflict-card-head > div {
  min-width: 0;
}

.conflict-type-pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 4px 9px;
  border-radius: 999px;
  background: #f1f5f9;
  color: #475569;
  font-size: 11px;
  font-weight: 800;
  white-space: nowrap;
}

.conflict-values-grid {
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 8px;
  margin: 12px 0 8px;
}

.conflict-value {
  min-width: 0;
  gap: 5px;
  padding: 9px;
  border: 1px solid var(--c-border);
  border-radius: 10px;
  background: #f8fafc;
}

.conflict-value strong {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 12.5px;
  line-height: 1.55;
}

.conflict-card p {
  margin: 0;
  color: var(--c-text-secondary);
  font-size: 12.5px;
  line-height: 1.6;
}

@media (max-width: 1100px) {
  .compare-page {
    width: min(100% - 24px, 1240px);
  }

  .comparison-table th,
  .comparison-table td,
  .tool-col {
    width: 220px;
    min-width: 220px;
    max-width: 220px;
  }
}

@media (max-width: 720px) {
  .compare-page {
    width: min(100% - 18px, 1240px);
  }

  .mobile-comparison-list {
    display: grid;
  }

  .mobile-tool-row {
    border: 1px solid var(--c-border);
    border-radius: 10px;
    padding: 10px;
    background: #f8fafc;
  }
}

</style>
