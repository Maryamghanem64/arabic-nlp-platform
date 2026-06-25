<template>
  <div class="page-wrap compare-page page-stack">
    <section class="hero-band compare-hero">
      <div class="hero-content">
        <span class="eyebrow">Comparative NLP dashboard</span>
        <h1 class="hero-title">Token-level comparison with live status-aware sections.</h1>
        <p class="hero-copy">
          The comparison table, conflict review, and fusion provenance are driven by the backend status snapshot.
          CompareView intentionally excludes SinaTools until token-level compare data exists.
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
          <button class="btn btn-subtle" @click="loadSample">Benchmark sample</button>
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
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="compare">
          {{ loading ? 'Running comparison...' : 'Run Comparative Analysis' }}
        </button>
        <button class="btn btn-secondary" :disabled="!hasResults" @click="copyResults">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

    <nav class="section-nav" aria-label="Compare sections">
      <a href="#evaluation-summary">Evaluation</a>
      <a href="#token-comparison">Comparison</a>
      <a href="#conflicts">Conflicts</a>
      <a href="#fusion-sources">Sources</a>
    </nav>

    <div v-if="statusError && !toolStatusesLoaded" class="error-state">
      <div>
        <strong>Backend status unavailable</strong>
        <p>{{ statusError.message || 'The comparison page could not load tool availability from GET /.' }}</p>
      </div>
    </div>

    <div v-if="loading" class="loading-grid">
      <div v-for="n in 4" :key="n" class="panel panel-pad skeleton-metric">
        <span class="skeleton"></span>
        <span class="skeleton wide"></span>
      </div>
    </div>

    <div v-if="error" class="error-state">
      <div>
        <strong>Comparison failed</strong>
        <p>{{ error }}</p>
        <button class="btn btn-secondary" @click="compare">Retry</button>
      </div>
    </div>

    <template v-if="hasResults && !loading">
      <section id="evaluation-summary" class="panel panel-pad section-block">
        <div class="section-head titled">
          <h2 class="section-title">
            <span class="section-kicker"><span class="section-num">01</span><span class="section-icon">E</span></span>
            Evaluation summary
          </h2>
        </div>

        <div class="evaluation-compact">
          <article class="metric-strip">
            <div class="metric-strip-head">
              <span class="metric-label">POS agreement</span>
              <strong>{{ evaluation.pos_agreement_pct || '0%' }}</strong>
            </div>
            <div class="progress-track"><span :style="{ width: `${metricPercent(evaluation.pos_agreement_pct)}%` }"></span></div>
          </article>

          <article class="metric-strip">
            <div class="metric-strip-head">
              <span class="metric-label">Lemma match</span>
              <strong>{{ evaluation.lemma_match_pct || '0%' }}</strong>
            </div>
            <div class="progress-track"><span :style="{ width: `${metricPercent(evaluation.lemma_match_pct)}%` }"></span></div>
          </article>

          <article class="metric-strip badge-strip">
            <span class="metric-label">POS F1</span>
            <span class="metric-badge">{{ formatDecimal(evaluation.pos_f1) }}</span>
          </article>

          <article class="metric-strip chips-strip">
            <span class="metric-label">Active tools</span>
            <div class="chip-row">
              <span v-for="tool in evaluation.active_tools || []" :key="tool" class="tool-chip tool-chip-active">{{ toolLabel(tool) }}</span>
            </div>
          </article>

          <article class="metric-strip chips-strip">
            <span class="metric-label">Excluded tools</span>
            <div class="chip-row">
              <span v-for="tool in evaluation.excluded_tools || []" :key="tool" class="tool-chip tool-chip-muted">{{ toolLabel(tool) }}</span>
            </div>
          </article>

          <p class="metrics-note"><em>{{ evaluation.metrics_note || 'No metrics note was returned by the backend.' }}</em></p>
        </div>
      </section>

      <section id="token-comparison" class="panel panel-pad section-block">
        <div class="section-head titled">
          <h2 class="section-title">
            <span class="section-kicker"><span class="section-num">02</span><span class="section-icon">T</span></span>
            Token comparison table
          </h2>
        </div>

        <div v-if="comparisonRows.length && compareCols.length" class="table-scroll compare-scroll">
          <table class="comparison-table">
            <thead>
              <tr>
                <th class="sticky-word-col">Word</th>
                <th v-for="toolKey in compareCols" :key="toolKey" class="tool-col">
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
                  <div v-if="row.tools[toolKey].available" class="cell-stack">
                    <span
                      v-for="line in row.tools[toolKey].lines"
                      :key="`${row.index}-${toolKey}-${line.label}`"
                      class="cell-line"
                      :class="{ arabic: line.rtl }"
                      :dir="line.rtl ? 'rtl' : null"
                      :lang="line.rtl ? 'ar' : null"
                    >
                      {{ line.value }}
                    </span>
                  </div>
                  <span v-else class="missing-cell">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty-state">No comparison rows were returned, or no compare-enabled tools are currently online.</div>
      </section>

      <section id="conflicts" class="panel panel-pad section-block">
        <div class="section-head titled">
          <h2 class="section-title">
            <span class="section-kicker"><span class="section-num">03</span><span class="section-icon">!</span></span>
            Conflicts
          </h2>
          <span :class="['pill', conflictBadge.className]">{{ conflictBadge.label }}</span>
        </div>

        <div v-if="conflictRows.length" class="conflict-list">
          <article v-for="conflict in conflictRows" :key="`${conflict.index}-${conflict.feature}-${conflict.key}`" class="conflict-card">
            <div class="conflict-grid">
              <div class="conflict-word arabic" dir="rtl" lang="ar">{{ conflict.word }}</div>
              <div class="conflict-feature">{{ conflict.feature }}</div>
              <div class="conflict-values">
                <span class="conflict-value">{{ conflict.toolA }} = {{ conflict.valueA }}</span>
                <span class="conflict-arrow">→</span>
                <span class="conflict-value">{{ conflict.toolB }} = {{ conflict.valueB }}</span>
              </div>
              <span :class="['pill', severityClass(conflict.severity)]">{{ conflict.severity }}</span>
            </div>
          </article>
        </div>
        <div v-else class="empty-state">No conflicts were returned by fusion.</div>
      </section>

      <section id="fusion-sources" class="panel panel-pad section-block">
        <div class="section-head titled">
          <h2 class="section-title">
            <span class="section-kicker"><span class="section-num">04</span><span class="section-icon">F</span></span>
            Fusion sources
          </h2>
        </div>

        <div v-if="fusionRows.length" class="fusion-list">
          <article v-for="row in fusionRows" :key="`fusion-${row.index}`" class="fusion-card">
            <div class="fusion-card-head">
              <strong class="arabic" dir="rtl" lang="ar">{{ row.word }}</strong>
              <span v-if="row.confidence" :class="['pill', confidenceClass(row.confidence)]">{{ row.confidence }}</span>
            </div>

            <table class="source-mini-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="source in fusionSourceRows(row.sources)" :key="`${row.index}-${source.feature}`">
                  <td>{{ source.feature }}</td>
                  <td>
                    <span class="source-chip" :style="{ backgroundColor: source.color }">
                      {{ source.shortLabel }}
                    </span>
                  </td>
                </tr>
                <tr v-if="!fusionSourceRows(row.sources).length">
                  <td colspan="2" class="empty-source-cell">—</td>
                </tr>
              </tbody>
            </table>
          </article>
        </div>
        <div v-else class="empty-state">Fusion sources are not available for this run.</div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import { API_BASE_URL, exportUrl } from '../api/nlpApi'
import { TOOL_CONFIG, TOOL_KEYS, toolOrder } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'
import { canonicalToken } from '../utils/tokenModel'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const comparisonPayload = ref(null)
const fusionPayload = ref(null)
const evaluationPayload = ref(null)
const copied = ref(false)

const { toolStatuses, activeTools, loading: statusLoading, error: statusError, refresh, toolStatus } = useToolStatus()

const compareEnabledTools = ['camel', 'farasa', 'stanza', 'qalsadi', 'alkhalil', 'arabert', 'udpipe', 'madamira']
const compareCols = computed(() => toolOrder(activeTools.value).filter((key) => compareEnabledTools.includes(key)))
const tokenEstimate = computed(() => (inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0))
const comparisonRows = computed(() => normalizeComparisonRows(comparisonPayload.value, compareCols.value))
const fusionRows = computed(() => normalizeFusionRows(fusionPayload.value))
const conflictRows = computed(() => normalizeConflicts(fusionRows.value.length ? fusionRows.value : comparisonRows.value))
const evaluation = computed(() => evaluationPayload.value?.evaluation || evaluationPayload.value || {})
const hasResults = computed(() => Boolean(comparisonRows.value.length || fusionRows.value.length || Object.keys(evaluation.value).length))
const toolStatusesLoaded = computed(() => Object.keys(toolStatuses.value).length > 0)
const conflictBadge = computed(() => {
  const count = conflictRows.value.length
  return count
    ? { label: `Conflicts (${count})`, className: 'pill-red' }
    : { label: 'No conflicts', className: 'pill-green' }
})
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
  fusionPayload.value = null
  evaluationPayload.value = null
  copied.value = false

  try {
    const [compareResult, fusionResult, evaluationResult] = await Promise.allSettled([
      axios.get(`${API_BASE_URL}/compare`, { params: { text: inputText.value, tools: compareCols.value.join(',') } }),
      axios.get(`${API_BASE_URL}/fusion`, { params: { text: inputText.value } }),
      axios.get(`${API_BASE_URL}/evaluate`, { params: { text: inputText.value } }),
    ])

    if (compareResult.status !== 'fulfilled') {
      throw new Error(readError(compareResult.reason, 'Unable to run comparison.'))
    }

    comparisonPayload.value = compareResult.value.data
    fusionPayload.value = fusionResult.status === 'fulfilled' ? fusionResult.value.data : null
    evaluationPayload.value = evaluationResult.status === 'fulfilled' ? evaluationResult.value.data : null
  } catch (e) {
    error.value = e.message || 'Failed to connect to the backend.'
  } finally {
    loading.value = false
  }
}

async function copyResults() {
  if (!hasResults.value) return
  const payload = JSON.stringify(
    {
      comparison: comparisonPayload.value,
      fusion: fusionPayload.value,
      evaluation: evaluationPayload.value,
    },
    null,
    2,
  )
  await navigator.clipboard.writeText(payload)
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}

function guardExport(event) {
  if (!hasResults.value) event.preventDefault()
}

function loadSample() {
  inputText.value = 'قرأ الطالب الكتب في المكتبة'
}

function clear() {
  inputText.value = ''
  comparisonPayload.value = null
  fusionPayload.value = null
  evaluationPayload.value = null
  error.value = ''
  copied.value = false
}

function normalizeComparisonRows(payload, cols) {
  const rows = Array.isArray(payload?.comparison) ? payload.comparison : []
  return rows.map((row, index) => {
    const tools = row?.tools || row || {}
    const normalizedTools = Object.fromEntries(
      cols.map((toolKey) => [toolKey, normalizeToolCell(toolKey, tools[toolKey])]),
    )

    return {
      index,
      word: row?.word || row?.surface || `#${index + 1}`,
      tools: normalizedTools,
      conflicts: Array.isArray(row?.conflicts) ? row.conflicts : [],
    }
  })
}

function normalizeToolCell(toolKey, raw) {
  const config = TOOL_CONFIG[toolKey] || {}
  const best = canonicalToken(raw)
  const lines = []

  for (const field of config.provides || []) {
    pushLine(lines, readField(best, field), field, ['lemma', 'root', 'gloss', 'stem'].includes(field))
  }

  if (best.normalized || best.note) {
    pushLine(lines, best.note || '(normalized)', 'note', true)
  }

  return {
    available: lines.length > 0,
    lines,
  }
}

function pushLine(target, value, label, rtl = false) {
  if (value === null || value === undefined || value === '') return
  target.push({
    label,
    value: Array.isArray(value) ? value.join(' + ') : String(value),
    rtl,
  })
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
    conflicts: Array.isArray(row?.conflicts) ? row.conflicts : [],
    confidence: row?.confidence || row?.final?.confidence_level || '',
  }))
}

function normalizeConflicts(rows) {
  return rows.flatMap((row) =>
    (row.conflicts || []).map((conflict, conflictIndex) => ({
      index: row.index,
      key: `${row.index}-${conflictIndex}`,
      word: row.word,
      feature: conflict.feature || 'feature',
      toolA: conflict.tool_a || conflict.toolA || 'tool_a',
      toolB: conflict.tool_b || conflict.toolB || 'tool_b',
      valueA: conflict.tool_a_value || conflict.value_a || conflict.toolAValue || conflict.tool_a || '—',
      valueB: conflict.tool_b_value || conflict.value_b || conflict.toolBValue || conflict.tool_b || '—',
      severity: String(conflict.severity || 'unknown').toLowerCase(),
    })),
  )
}

function unwrapAnalysis(raw) {
  return canonicalToken(raw)
}

function dependencyLabel(raw) {
  const dependency = raw?.dependency || raw?.dep || {}
  const deprel = dependency.deprel || raw?.deprel || ''
  if (!deprel) return ''
  return dependency.head_text ? `${deprel} -> ${dependency.head_text}` : String(deprel)
}

function readField(raw, field) {
  if (!raw) return ''
  if (field === 'pos') return raw.pos || raw.upos || ''
  if (field === 'lemma') return raw.lemma || ''
  if (field === 'root') return raw.root || ''
  if (field === 'segmentation') return Array.isArray(raw.segmentation) ? raw.segmentation.join(' + ') : raw.segmentation || ''
  if (field === 'dependency') return dependencyLabel(raw)
  if (field === 'case') return raw.case || ''
  if (field === 'gloss') return raw.gloss || ''
  if (field === 'stem') return raw.stem || ''
  return raw[field] || ''
}

function fusionSourceRows(sources) {
  return Object.entries(sources || {}).map(([feature, tool]) => ({
    feature,
    tool,
    shortLabel: shortToolLabel(tool),
    color: TOOL_CONFIG[tool]?.color || '#64748b',
  }))
}

function shortToolLabel(toolKey) {
  const label = TOOL_CONFIG[toolKey]?.label || toolKey || 'unknown'
  return String(label).split('/')[0].trim().split(' ')[0]
}

function toolLabel(toolKey) {
  return TOOL_CONFIG[toolKey]?.label || toolKey
}

function confidenceClass(confidence) {
  const value = String(confidence || '').toLowerCase()
  if (value === 'high') return 'pill-green'
  if (value === 'medium') return 'pill-blue'
  if (value === 'low') return 'pill-red'
  return 'pill-gray'
}

function severityClass(severity) {
  if (severity === 'high') return 'pill-red'
  if (severity === 'medium') return 'pill-amber'
  return 'pill-gray'
}

function metricPercent(value) {
  const parsed = Number.parseFloat(String(value || '').replace('%', ''))
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0
}

function formatDecimal(value) {
  return typeof value === 'number' ? value.toFixed(3) : '0.000'
}

function toArray(value) {
  if (Array.isArray(value)) return value.filter(Boolean)
  if (typeof value === 'string' && value.trim()) return value.split(/\s+/).filter(Boolean)
  return []
}

function readError(reason, fallback) {
  return reason?.response?.data?.detail || reason?.response?.data?.error || reason?.message || fallback
}

function value(item) {
  return item || '—'
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
  font-weight: 850;
}

.disabled {
  pointer-events: none;
  opacity: 0.5;
}

.section-nav {
  position: sticky;
  top: 12px;
  z-index: 8;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 16px 0 18px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(8px);
}

.section-nav a {
  padding: 8px 12px;
  border-radius: 999px;
  color: var(--navy);
  background: #eef5ff;
  font-size: 13px;
  font-weight: 850;
}

.section-block {
  scroll-margin-top: 92px;
}

.section-head.titled {
  align-items: center;
}

.section-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-right: 8px;
}

.section-num,
.section-icon {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 950;
}

.section-num {
  background: #dbeafe;
  color: #1d4ed8;
}

.section-icon {
  background: #eef2ff;
  color: #4338ca;
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

.evaluation-compact {
  display: grid;
  gap: 12px;
}

.metric-strip {
  display: grid;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfdff;
}

.metric-strip-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-strip strong {
  color: var(--navy);
  font-size: 18px;
  font-weight: 950;
}

.progress-track {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: #e7edf5;
}

.progress-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--cyan), var(--blue));
}

.badge-strip {
  width: fit-content;
}

.metric-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 64px;
  min-height: 34px;
  padding: 6px 10px;
  border-radius: 8px;
  background: #eef6ff;
  color: var(--navy);
  font-size: 15px;
  font-weight: 950;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.tool-chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  color: #fff;
  font-size: 12px;
  font-weight: 900;
}

.tool-chip-active {
  background: #15803d;
}

.tool-chip-muted {
  background: #94a3b8;
}

.metrics-note {
  margin: 0;
  color: var(--muted);
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
  width: 130px;
}

.comparison-table .sticky-word-col {
  position: sticky;
  left: 0;
  z-index: 2;
  width: 120px;
  min-width: 120px;
  max-width: 120px;
  background: #fff;
}

.comparison-table thead .sticky-word-col {
  z-index: 4;
}

.comparison-table thead th {
  position: sticky;
  top: 0;
  z-index: 3;
  background: #f8fafc;
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
  font-weight: 950;
}

.word-cell .mono {
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}

.tool-col {
  min-width: 130px;
  max-width: 130px;
  vertical-align: top;
}

.cell-stack {
  display: grid;
  gap: 3px;
}

.cell-line {
  display: block;
  overflow: hidden;
  color: var(--ink);
  font-size: 13px;
  font-weight: 850;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cell-line.arabic {
  font-size: 14px;
}

.missing-cell {
  color: var(--muted);
  font-size: 18px;
  font-weight: 800;
}

.conflict-list {
  display: grid;
  gap: 12px;
}

.conflict-card {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}

.conflict-grid {
  display: grid;
  grid-template-columns: 140px minmax(0, 160px) minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
}

.conflict-word {
  font-size: 18px;
  font-weight: 950;
}

.conflict-feature {
  padding: 6px 10px;
  border-radius: 999px;
  background: #eef2f7;
  color: #334155;
  font-size: 12px;
  font-weight: 900;
}

.conflict-values {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.conflict-value {
  padding: 6px 10px;
  border-radius: 999px;
  background: #f8fafc;
  color: var(--ink);
  font-size: 12px;
  font-weight: 850;
}

.conflict-arrow {
  color: var(--muted);
  font-weight: 900;
}

.fusion-list {
  display: grid;
  gap: 12px;
}

.fusion-card {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfdff;
}

.fusion-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.source-mini-table {
  width: 100%;
  min-width: 320px;
  border-collapse: collapse;
}

.source-mini-table th,
.source-mini-table td {
  padding: 8px 10px;
  text-align: left;
  border-top: 1px solid #e5e7eb;
}

.source-mini-table thead th {
  border-top: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.source-chip {
  display: inline-flex;
  align-items: center;
  padding: 5px 9px;
  border-radius: 999px;
  color: #fff;
  font-size: 12px;
  font-weight: 900;
}

.empty-source-cell {
  color: var(--muted);
  text-align: center;
}

.empty-state {
  padding: 16px;
  border: 1px dashed #cbd5e1;
  border-radius: 10px;
  color: var(--muted);
  background: #fafcff;
}

.section-block {
  scroll-margin-top: 90px;
}

@media (max-width: 1100px) {
  .loading-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .conflict-grid {
    grid-template-columns: 1fr;
  }

  .comparison-table th,
  .comparison-table td {
    width: 130px;
  }
}

@media (max-width: 720px) {
  .loading-grid {
    grid-template-columns: 1fr;
  }

  .section-nav {
    position: static;
  }
}
</style>
