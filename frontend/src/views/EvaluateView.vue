<template>
  <div class="page-wrap eval-page page-stack">
    <section class="hero-band eval-hero">
      <div class="hero-content">
        <span class="eyebrow">Capability-aware evaluation</span>
        <h1 class="hero-title">Report observed agreement without claiming gold-standard accuracy.</h1>
        <p class="hero-copy">
          The page evaluates comparable analyzer evidence only. Unsupported capabilities are excluded from metric
          denominators, and every score is interpreted as consistency between tools rather than correctness.
        </p>
        <p class="page-note">No AI-generated interpretation is used. Metrics are derived directly from analyzer outputs.</p>
        <p class="page-note">
          Metrics measure agreement or evidence coverage among comparable analyzer outputs. They do not represent gold-standard linguistic accuracy.
        </p>
      </div>
    </section>

    <section class="panel panel-pad input-section">
      <div class="section-head">
        <div>
          <h2 class="section-title">Arabic input</h2>
          <p class="section-subtitle">Run the backend evaluation pipeline on one sentence.</p>
        </div>
        <span class="method-chip">Agreement ≠ accuracy</span>
      </div>

      <div class="input-row">
        <textarea
          id="eval-input"
          v-model="inputText"
          class="arabic-input"
          placeholder="مثال: قرأ الطالب الكتب في المكتبة"
          rows="2"
          dir="rtl"
          lang="ar"
        ></textarea>

        <button class="run-btn" :disabled="loading || !inputText.trim()" @click="runEvaluation">
          {{ loading ? 'Computing...' : 'Run evaluation' }}
        </button>
      </div>

      <div class="examples-row">
        <span class="examples-label">Examples:</span>
        <button
          v-for="ex in EXAMPLE_SENTENCES"
          :key="ex"
          class="example-chip"
          type="button"
          @click="runExample(ex)"
        >
          {{ ex }}
        </button>
      </div>
    </section>

    <section class="evaluation-scope-strip">
      <article v-for="item in scopeItems" :key="item.title" class="scope-card">
        <span>{{ item.kicker }}</span>
        <strong>{{ item.title }}</strong>
        <p>{{ item.text }}</p>
      </article>
    </section>

    <details class="panel panel-pad capability-details">
      <summary>
        <span>
          <strong>Capability matrix</strong>
          <small>Eligibility rules used before metrics are interpreted.</small>
        </span>
        <span class="method-chip">Open methodology scope</span>
      </summary>

      <div class="capability-details-body">
        <div class="section-head">
          <div>
            <h2 class="section-title">Evaluation scope</h2>
            <p class="section-subtitle">Each analyzer is evaluated only on the capabilities it can reasonably provide.</p>
          </div>
        </div>

        <div class="capability-table-wrap">
          <table class="capability-table">
            <thead>
              <tr>
                <th>Analyzer</th>
                <th v-for="feature in capabilityFeatures" :key="feature">{{ feature }}</th>
                <th>Research role</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in capabilityMatrix" :key="row.key">
                <td><ToolBadge :tool="row.key" /></td>
                <td v-for="feature in capabilityFeatures" :key="feature">
                  <span :class="['cap-cell', capabilityClass(row[feature])]">{{ row[feature] }}</span>
                </td>
                <td class="role-cell">{{ row.role }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p class="scope-footnote">
          SinaTools is listed as an optional local-resource analyzer. It contributes only when its local resources are
          available. AraBERT is contextual evidence and is not treated as a direct morphology-table competitor.
        </p>
      </div>
    </details>

    <div v-if="error" class="error-banner">
      <strong>Evaluation failed</strong>
      <span>{{ error }}</span>
    </div>

    <div v-if="loading" class="loading-state eval-loading">
      <span class="spinner--dark" aria-hidden="true"></span>
      <p>Computing capability-scoped agreement evidence...</p>
    </div>

    <template v-if="evalResult && !loading">
      <section class="panel panel-pad run-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Run participation</h2>
            <p class="section-subtitle">Tools reported by the backend for this evaluation request.</p>
          </div>
          <span class="runtime-chip">{{ requestDuration }} ms request</span>
        </div>

        <div class="participation-grid">
          <article class="report-card">
            <span class="report-label">Participating tools</span>
            <div class="tool-row">
              <ToolBadge v-for="tool in activeTools" :key="tool" :tool="tool" />
              <span v-if="!activeTools.length" class="null-value">No participating tools returned.</span>
            </div>
          </article>

          <article class="report-card">
            <span class="report-label">Excluded / unavailable</span>
            <div class="excluded-list">
              <span v-for="tool in excludedTools" :key="tool">{{ tool }}</span>
              <span v-if="!excludedTools.length" class="null-value">None reported</span>
            </div>
          </article>

          <article class="report-card">
            <span class="report-label">Degraded runtime notes</span>
            <div class="excluded-list">
              <span v-for="note in degradedNotes" :key="note">{{ note }}</span>
              <span v-if="!degradedNotes.length" class="null-value">None reported</span>
            </div>
          </article>
        </div>
      </section>

      <section class="metrics-section">
        <article v-for="metric in metricCards" :key="metric.key" class="kpi-card">
          <span class="kpi-label">{{ metric.label }}</span>
          <strong class="kpi-value" :class="getScoreClass(metric.value)">
            {{ percentLabel(metric.value, metric.digits) }}
          </strong>
          <span class="evaluated-count">Evaluated tokens: {{ metric.count ?? 0 }}</span>
          <div class="metric-tool-block">
            <span>Metric contributors</span>
            <div class="tool-row compact-tool-row">
              <ToolBadge v-for="tool in metric.contributors" :key="`${metric.key}-metric-${tool}`" :tool="tool" />
              <span v-if="!metric.contributors.length" class="null-value">No comparable values returned.</span>
            </div>
          </div>
          <div class="metric-tool-block">
            <span>Capability contributors</span>
            <div class="tool-row compact-tool-row">
              <ToolBadge v-for="tool in metric.capability" :key="`${metric.key}-cap-${tool}`" :tool="tool" />
              <span v-if="!metric.capability.length" class="null-value">No capable active tools reported.</span>
            </div>
          </div>
          <p class="kpi-note">{{ metric.note }}</p>
        </article>
      </section>

      <section class="panel panel-pad contributor-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Contributor distinction</h2>
            <p class="section-subtitle">Capability contributors are tools that can support a feature. Metric contributors are tools that returned comparable values in this run.</p>
          </div>
        </div>
      </section>

      <ScientificChart
        type="bar"
        title="Observed Agreement Metrics"
        subtitle="Separate capability-scoped indicators; no composite quality score is computed."
        badge="Observed evidence"
        :labels="observedMetrics.labels"
        :datasets="observedMetrics.datasets"
        :height="300"
        aria-label="Observed agreement metrics"
        empty-title="No evaluation metrics"
        empty-text="Run evaluation to populate capability-scoped indicators."
      />

      <section class="panel panel-pad report-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Disagreement evidence and interpretation boundary</h2>
            <p class="section-subtitle">Backend conflict rows are preserved without assigning a gold answer.</p>
          </div>
          <a class="methodology-link" href="/docs/evaluation_methodology.md" target="_blank" rel="noreferrer">
            Evaluation methodology
          </a>
        </div>

        <div class="report-grid">
          <article class="report-card">
            <span class="report-label">Observed disagreements</span>

            <div class="conflict-mini-grid">
              <div v-for="(conflict, index) in allConflicts" :key="index" class="conflict-card">
                <strong class="arabic-value" dir="rtl" lang="ar">
                  {{ conflict.word || conflict.token || `#${index + 1}` }}
                </strong>
                <span class="disagreement-badge">{{ conflict.feature || 'feature' }}</span>
                <span class="conflict-text">{{ conflictText(conflict) }}</span>
              </div>

              <span v-if="!allConflicts.length" class="null-value">No observed disagreements were reported.</span>
            </div>
          </article>

          <article class="report-card interpretation-card">
            <span class="report-label">Interpretation boundary</span>
            <p class="metrics-note">
              {{ rawNote || 'Metrics are capability-aware. Each score is computed only over tools that support the evaluated linguistic feature. Agreement indicates consistency of output, not correctness against a gold standard.' }}
            </p>

            <ul class="boundary-list">
              <li>High agreement can still be wrong without gold labels.</li>
              <li>Low agreement can reflect valid analyzer convention differences.</li>
              <li>Unsupported features are excluded rather than penalized.</li>
            </ul>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import ToolBadge from '@/components/badges/ToolBadge.vue'
import ScientificChart from '@/components/charts/ScientificChart.vue'
import { evaluateText } from '@/api/nlpApi'
import { recordAnalysis } from '@/utils/analysisHistory'
import { formatToolList } from '@/utils/researchSemantics'

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const evalResult = ref(null)
const activeTools = ref([])
const excludedTools = ref([])
const degradedNotes = ref([])
const capabilityContributors = ref({})
const metricContributors = ref({})
const evaluatedTokenCounts = ref({})
const allConflicts = ref([])
const posConflicts = ref([])
const rawNote = ref('')
const requestDuration = ref(0)
const runId = ref(0)

const EXAMPLE_SENTENCES = [
  'قرأ الطالب الكتب في المكتبة',
  'وجدت المعلمة طالبة مجتهدة في الفصل',
  'يكتب الصحفي المقالة كل يوم',
]

const scopeItems = [
  {
    kicker: 'Scope',
    title: 'Capability-first denominator',
    text: 'A metric only includes tools that support the evaluated feature.',
  },
  {
    kicker: 'Interpretation',
    title: 'Agreement is consistency',
    text: 'The score does not claim correctness without labeled gold data.',
  },
  {
    kicker: 'Evidence',
    title: 'Conflicts remain inspectable',
    text: 'Disagreements are preserved as evidence rather than hidden behind one score.',
  },
]

const capabilityFeatures = ['segmentation', 'lemma', 'root', 'pos', 'morphology', 'dependency']

const capabilityMatrix = [
  {
    key: 'farasa',
    segmentation: 'Strong',
    lemma: 'N/A',
    root: 'N/A',
    pos: 'N/A',
    morphology: 'N/A',
    dependency: 'N/A',
    role: 'Segmentation anchor',
  },
  {
    key: 'camel',
    segmentation: 'N/A',
    lemma: 'Strong',
    root: 'Strong',
    pos: 'Strong',
    morphology: 'Strong',
    dependency: 'N/A',
    role: 'Primary morphology evidence',
  },
  {
    key: 'sinatools',
    segmentation: 'N/A',
    lemma: 'Strong',
    root: 'Supported',
    pos: 'Strong',
    morphology: 'Supported',
    dependency: 'N/A',
    role: 'Optional lexical-morphology evidence',
  },
  {
    key: 'alkhalil',
    segmentation: 'N/A',
    lemma: 'Supported',
    root: 'Supported',
    pos: 'Supported',
    morphology: 'Strong',
    dependency: 'N/A',
    role: 'Rule-based morphology support',
  },
  {
    key: 'stanza',
    segmentation: 'N/A',
    lemma: 'Supported',
    root: 'N/A',
    pos: 'Strong',
    morphology: 'Supported',
    dependency: 'Strong',
    role: 'UD syntax expert',
  },
  {
    key: 'udpipe',
    segmentation: 'N/A',
    lemma: 'Supported',
    root: 'N/A',
    pos: 'Strong',
    morphology: 'Supported',
    dependency: 'Strong',
    role: 'Independent UD syntax support',
  },
  {
    key: 'qalsadi',
    segmentation: 'N/A',
    lemma: 'Supported',
    root: 'N/A',
    pos: 'N/A',
    morphology: 'Partial',
    dependency: 'N/A',
    role: 'Rule-based lexical support',
  },
  {
    key: 'arabert',
    segmentation: 'N/A',
    lemma: 'N/A',
    root: 'N/A',
    pos: 'N/A',
    morphology: 'N/A',
    dependency: 'N/A',
    role: 'Contextual representation; outside direct feature agreement',
  },
]

const metricCards = computed(() => [
  {
    key: 'pos',
    label: 'POS Agreement',
    value: evalResult.value?.pos_agreement,
    digits: 1,
    count: evaluatedTokenCounts.value.pos,
    contributors: metricContributors.value.pos || [],
    capability: capabilityContributors.value.pos || [],
    note: 'Agreement among comparable POS values returned in this run.',
  },
  {
    key: 'lemma',
    label: 'Lemma Match',
    value: evalResult.value?.lemma_match,
    digits: 1,
    count: evaluatedTokenCounts.value.lemma,
    contributors: metricContributors.value.lemma || [],
    capability: capabilityContributors.value.lemma || [],
    note: 'Match among comparable lemma evidence after backend normalization.',
  },
  {
    key: 'root',
    label: 'Root Agreement',
    value: evalResult.value?.root_agreement,
    digits: 1,
    count: evaluatedTokenCounts.value.root,
    contributors: metricContributors.value.root || [],
    capability: capabilityContributors.value.root || [],
    note: 'Root scoring excludes tokens where root comparison is not linguistically meaningful.',
  },
  {
    key: 'segmentation',
    label: 'Segmentation Coverage',
    value: evalResult.value?.segmentation_coverage,
    digits: 0,
    count: evaluatedTokenCounts.value.segmentation,
    contributors: metricContributors.value.segmentation || [],
    capability: capabilityContributors.value.segmentation || [],
    note: 'Coverage of returned segmentation evidence among capable tools.',
  },
])

const observedMetrics = computed(() => ({
  labels: metricCards.value.map((metric) => metric.label),
  datasets: [
    {
      label: 'Observed %',
      data: metricCards.value.map((metric) => toPercent(metric.value)),
      backgroundColor: '#315C8C',
    },
  ],
}))

async function runEvaluation() {
  if (!inputText.value.trim()) return

  const currentRunId = ++runId.value
  loading.value = true
  error.value = ''
  evalResult.value = null
  activeTools.value = []
  excludedTools.value = []
  degradedNotes.value = []
  capabilityContributors.value = {}
  metricContributors.value = {}
  evaluatedTokenCounts.value = {}
  allConflicts.value = []
  posConflicts.value = []
  rawNote.value = ''
  requestDuration.value = 0

  try {
    const started = performance.now()
    const data = await evaluateText(inputText.value)
    if (currentRunId !== runId.value) return
    requestDuration.value = Math.round(performance.now() - started)

    const source = data?.data || data
    const raw = source?.evaluation || source

    evalResult.value = {
      pos_agreement: toScalar(raw.pos_agreement ?? raw.pos_agreement_pct),
      lemma_match: toScalar(
        raw.lemma_normalized_match ??
          raw.lemma_normalized_match_pct ??
          raw.lemma_match ??
          raw.lemma_match_pct,
      ),
      root_agreement: toScalar(raw.root_agreement ?? raw.root_agreement_pct),
      segmentation_coverage: toScalar(raw.segmentation_coverage),
    }

    activeTools.value = normalizeToolList(raw.active_tools || source.active_tools || [])
    excludedTools.value = normalizeToolList(raw.excluded_tools || source.excluded_tools || [])
    degradedNotes.value = normalizeToolList(raw.degraded_notes || [])
    capabilityContributors.value = normalizeContributorMap(raw.capability_contributors)
    metricContributors.value = normalizeContributorMap(raw.metric_contributors)
    evaluatedTokenCounts.value = raw.evaluated_token_counts || {}
    allConflicts.value = Array.isArray(raw.all_conflicts) ? raw.all_conflicts : []
    posConflicts.value = Array.isArray(raw.pos_conflicts)
      ? raw.pos_conflicts
      : Array.isArray(raw.all_conflicts)
        ? raw.all_conflicts
        : []
    rawNote.value = raw.metrics_note || ''

    recordAnalysis({
      page: 'Evaluate',
      text: inputText.value.trim(),
      summary: `${toPercent(evalResult.value.pos_agreement)}% POS agreement | capability-aware`,
    })
  } catch (e) {
    if (currentRunId !== runId.value) return
    error.value =
      e?.response?.data?.detail ||
      e?.response?.data?.error ||
      e?.message ||
      'Unable to connect to the evaluation service.'
  } finally {
    if (currentRunId === runId.value) loading.value = false
  }
}

function runExample(example) {
  inputText.value = example
  runEvaluation()
}

function normalizeToolList(value) {
  return formatToolList(value)
}

function normalizeContributorMap(value) {
  if (!value || typeof value !== 'object') return {}
  return Object.fromEntries(
    Object.entries(value).map(([feature, tools]) => [feature, normalizeToolList(tools)]),
  )
}

function capabilityClass(value) {
  if (value === 'Strong') return 'cap-strong'
  if (value === 'Supported') return 'cap-supported'
  if (value === 'Partial') return 'cap-partial'
  return 'cap-na'
}

function toScalar(value) {
  if (value == null) return 0
  if (typeof value === 'number') return value > 1 ? value / 100 : value
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value.replace('%', ''))
    return Number.isFinite(parsed) ? (parsed > 1 ? parsed / 100 : parsed) : 0
  }
  return 0
}

function toPercent(value) {
  return Math.round(toScalar(value) * 100)
}

function percentLabel(score, digits = 1) {
  return `${(toScalar(score) * 100).toFixed(digits)}%`
}

function getScoreClass(score) {
  const value = toScalar(score)
  if (value >= 0.85) return 'score-high'
  if (value >= 0.6) return 'score-medium'
  return 'score-low'
}

function conflictText(conflict) {
  if (conflict?.tool_a || conflict?.tool_b) {
    const toolA = conflict.tool_a || 'Tool A'
    const toolB = conflict.tool_b || 'Tool B'
    const valueA = conflict.value_a ?? conflict.tool_a_value ?? conflict.raw_value_a ?? 'not returned'
    const valueB = conflict.value_b ?? conflict.tool_b_value ?? conflict.raw_value_b ?? 'not returned'
    const severity = conflict.severity ? ` / Severity: ${conflict.severity}` : ''
    return `${toolA}: ${valueA} / ${toolB}: ${valueB}${severity}`
  }

  if (conflict?.camel !== undefined || conflict?.stanza !== undefined) {
    return `CAMeL: ${conflict.camel ?? 'N/A'} / Stanza: ${conflict.stanza ?? 'N/A'}`
  }

  if (conflict?.values && typeof conflict.values === 'object') {
    return Object.entries(conflict.values)
      .map(([tool, value]) => `${tool}: ${value}`)
      .join(' / ')
  }

  return conflict?.message || conflict?.detail || 'Analyzer outputs differ for this feature.'
}
</script>

<style scoped>
.eval-page {
  width: min(96vw, 1500px);
}

.eval-hero {
  min-height: 250px;
}

.page-note,
.scope-footnote {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--c-accent-border);
  border-left: 3px solid var(--c-accent);
  border-radius: 10px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 13px;
  line-height: 1.55;
}

.input-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: stretch;
}

.arabic-input {
  min-height: 86px;
  padding: 12px 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-control);
  color: var(--c-text-primary);
  background: var(--c-surface);
  font-size: 20px;
  line-height: 1.7;
  resize: vertical;
  outline: none;
}

.arabic-input:focus {
  border-color: var(--c-accent);
  box-shadow: 0 0 0 3px var(--c-accent-light);
}

.run-btn {
  min-width: 160px;
  border-radius: 8px;
  background: var(--c-accent);
  color: #fff;
  font-weight: 700;
  padding: 0 18px;
  cursor: pointer;
}

.run-btn:disabled {
  opacity: .55;
  cursor: not-allowed;
}

.examples-row,
.tool-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.examples-label {
  color: var(--c-text-muted);
  font-size: 12px;
  align-self: center;
}

.example-chip {
  border: 1px solid var(--c-border);
  border-radius: 999px;
  background: var(--c-page-bg);
  color: var(--c-text-secondary);
  cursor: pointer;
  padding: 6px 11px;
}

.method-chip,
.runtime-chip {
  padding: 6px 10px;
  border: 1px solid var(--c-accent-border);
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 12px;
  font-weight: 700;
}

.evaluation-scope-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.scope-card {
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.scope-card span {
  display: block;
  margin-bottom: 6px;
  color: var(--c-text-muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .07em;
  text-transform: uppercase;
}

.scope-card strong {
  display: block;
  color: var(--c-text-primary);
  font-size: 15px;
}

.scope-card p {
  margin: 7px 0 0;
  color: var(--c-text-secondary);
  font-size: 12.5px;
  line-height: 1.6;
}

.capability-details summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  cursor: pointer;
  list-style: none;
}

.capability-details summary::-webkit-details-marker {
  display: none;
}

.capability-details summary span:first-child {
  display: grid;
  gap: 3px;
}

.capability-details summary small {
  color: var(--c-text-secondary);
  font-weight: 400;
}

.capability-details-body {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid var(--c-border);
}

.capability-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--c-border);
  border-radius: 12px;
  background: var(--c-surface);
}

.capability-table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
}

.capability-table th,
.capability-table td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--c-border);
  text-align: left;
  white-space: nowrap;
}

.capability-table th {
  background: #f8fafc;
  color: var(--c-text-secondary);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.cap-cell {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.cap-strong {
  background: var(--c-agreement-bg);
  color: var(--c-agreement-text);
}

.cap-supported {
  background: var(--c-accent-light);
  color: var(--c-accent-text);
}

.cap-partial {
  background: var(--c-warning-bg);
  color: var(--c-warning-text);
}

.cap-na {
  background: var(--c-na-bg);
  color: var(--c-na-text);
}

.role-cell {
  min-width: 260px;
  color: var(--c-text-secondary);
  white-space: normal;
}

.error-banner {
  display: grid;
  gap: 4px;
  padding: 12px 16px;
  border: 1px solid #e7b0b0;
  border-radius: var(--radius-control);
  color: #7e3f3f;
  background: #fbf2f2;
}

.eval-loading {
  min-height: 160px;
}

.participation-grid,
.report-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.report-card {
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 12px;
  background: var(--c-page-bg);
}

.report-label {
  display: block;
  margin-bottom: 10px;
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .07em;
}

.excluded-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.excluded-list span:not(.null-value) {
  padding: 5px 8px;
  border-radius: 999px;
  background: var(--c-na-bg);
  color: var(--c-na-text);
  font-size: 12px;
}

.metrics-section {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.kpi-card {
  display: grid;
  gap: 7px;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.kpi-label {
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.kpi-value {
  font-size: 1.55rem;
  font-weight: 800;
}

.score-high {
  color: var(--c-agreement-text);
}

.score-medium {
  color: var(--c-warning-text);
}

.score-low {
  color: var(--c-conflict-text);
}

.kpi-note {
  margin: 0;
  color: var(--c-text-secondary);
  font-size: 12px;
  line-height: 1.55;
}

.evaluated-count {
  color: var(--c-text-secondary);
  font-size: 12px;
  font-weight: 650;
}

.metric-tool-block {
  display: grid;
  gap: 6px;
  padding-top: 4px;
}

.metric-tool-block > span {
  color: var(--c-text-muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.compact-tool-row {
  margin-top: 0;
}

.contributor-panel {
  border-left: 3px solid var(--c-accent);
}

.conflict-mini-grid {
  display: grid;
  gap: 8px;
}

.conflict-card {
  display: grid;
  grid-template-columns: minmax(80px, auto) auto minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  padding: 10px;
  border: 1px solid #d9e2ec;
  border-radius: 10px;
  background: #fff;
}

.disagreement-badge {
  padding: 4px 8px;
  border-radius: 999px;
  background: #f1e5ca;
  color: #7a5a2e;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.conflict-text {
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--c-text-secondary);
  font-size: 12.5px;
}

.metrics-note {
  margin: 0;
  color: var(--c-text-secondary);
  line-height: 1.7;
}

.boundary-list {
  margin: 14px 0 0;
  padding-left: 18px;
  color: var(--c-text-secondary);
  line-height: 1.65;
}

.methodology-link {
  color: var(--c-accent-text);
  font-weight: 700;
}

@media (max-width: 1000px) {
  .eval-page {
    width: min(100% - 24px, 1240px);
  }

  .evaluation-scope-strip,
  .metrics-section {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .participation-grid,
  .report-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .input-row,
  .evaluation-scope-strip,
  .metrics-section {
    grid-template-columns: 1fr;
  }

  .run-btn {
    min-height: 46px;
  }

  .capability-details summary {
    align-items: flex-start;
    flex-direction: column;
  }

  .conflict-card {
    grid-template-columns: 1fr;
  }
}
</style>
