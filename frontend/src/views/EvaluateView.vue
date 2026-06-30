<template>
  <div class="page-wrap eval-page page-stack">
    <section class="hero-band eval-hero">
      <div class="hero-content">
        <span class="eyebrow">Evaluation report</span>
        <h1 class="hero-title">Scientific tool agreement and coverage analysis.</h1>
        <p class="hero-copy">
          The page reports direct analyzer-derived metrics for agreement, coverage, active tools, and runtime.
        </p>
        <p class="page-note">
          Results shown on this page are computed directly from analyzer outputs. No AI-generated interpretation is used.
        </p>
      </div>
    </section>

    <section class="panel panel-pad input-section">
      <div class="section-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
          <p class="section-subtitle">The backend runs its evaluation pipeline against the current sentence.</p>
        </div>
      </div>

      <div class="input-row">
        <textarea
          id="eval-input"
          v-model="inputText"
          class="arabic-input"
          placeholder="Example: \u0642\u0631\u0623 \u0627\u0644\u0637\u0627\u0644\u0628 \u0627\u0644\u0643\u062a\u0628 \u0641\u064a \u0627\u0644\u0645\u0643\u062a\u0628\u0629"
          rows="2"
          dir="rtl"
          lang="ar"
        ></textarea>
        <button class="run-btn" :disabled="loading || !inputText.trim()" @click="runEvaluation">
          {{ loading ? 'Running evaluation...' : 'Run evaluation' }}
        </button>
      </div>

      <div class="examples-row">
        <button v-for="ex in EXAMPLE_SENTENCES" :key="ex" class="example-chip" @click="runExample(ex)">
          {{ ex }}
        </button>
      </div>
    </section>

    <div v-if="error" class="error-banner">{{ error }}</div>

    <div v-if="loading" class="loading-state eval-loading">
      <span class="spinner--dark"></span>
      <p>Computing agreement metrics and coverage...</p>
    </div>

    <section v-if="evalResult && !loading" class="metrics-section">
      <div class="kpi-grid">
        <article class="kpi-card">
          <div class="kpi-label">POS agreement</div>
          <div class="kpi-value" :class="getScoreClass(evalResult.pos_agreement)">
            {{ percentLabel(evalResult.pos_agreement, 1) }}
          </div>
          <div class="kpi-note">Across the active analyzers</div>
        </article>

        <article class="kpi-card">
          <div class="kpi-label">Normalized lemma match</div>
          <div class="kpi-value" :class="getScoreClass(evalResult.lemma_normalized_match)">
            {{ percentLabel(evalResult.lemma_normalized_match, 1) }}
          </div>
          <div class="kpi-note">After diacritic and orthographic normalization</div>
        </article>

        <article class="kpi-card">
          <div class="kpi-label">Exact lemma match</div>
          <div class="kpi-value" :class="getScoreClass(evalResult.lemma_exact_match)">
            {{ percentLabel(evalResult.lemma_exact_match, 1) }}
          </div>
          <div class="kpi-note">Strict token-level match</div>
        </article>

        <article class="kpi-card">
          <div class="kpi-label">Segmentation coverage</div>
          <div class="kpi-value" :class="getScoreClass(evalResult.segmentation_coverage)">
            {{ percentLabel(evalResult.segmentation_coverage, 0) }}
          </div>
          <div class="kpi-note">Farasa segmentation availability</div>
        </article>
      </div>

      <section class="analysis-visual-grid">
        <ScientificChart
          type="radar"
          title="Overall Evaluation"
          subtitle="Agreement, lemma, and coverage values."
          badge="Score"
          :labels="overallRadar.labels"
          :datasets="overallRadar.datasets"
          :height="280"
          aria-label="Overall evaluation radar chart"
          empty-title="No evaluation data"
          empty-text="Run the evaluation pipeline to populate the score radar."
        />

        <ScientificChart
          type="bar"
          title="Performance Snapshot"
          subtitle="Agreement and coverage at a glance."
          badge="Metrics"
          :labels="performanceBar.labels"
          :datasets="performanceBar.datasets"
          :height="280"
          aria-label="Performance snapshot chart"
          empty-title="No performance data"
          empty-text="The metrics chart appears after evaluation completes."
        />

        <ScientificChart
          type="bar"
          title="Runtime and Scale"
          subtitle="Frontend-measured response time and token volume."
          badge="Timing"
          :labels="runtimeBar.labels"
          :datasets="runtimeBar.datasets"
          :height="280"
          aria-label="Runtime chart"
          empty-title="No runtime data"
          empty-text="A completed request is required to show runtime metrics."
        />
      </section>

      <section class="panel panel-pad report-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Metrics Summary</h2>
            <p class="section-subtitle">Direct counts and availability indicators from the current run.</p>
          </div>
          <a class="methodology-link" href="/docs/evaluation_methodology.md" target="_blank" rel="noreferrer">
            Evaluation methodology
          </a>
        </div>

        <div class="report-grid">
          <article class="report-card">
            <span class="report-label">Active tools</span>
            <div class="tool-row">
              <ToolBadge v-for="tool in activeTools" :key="tool" :tool="tool" />
              <span v-if="!activeTools.length" class="null-value">No active tools were returned.</span>
            </div>
          </article>

          <article class="report-card">
            <span class="report-label">Excluded tools</span>
            <div class="excluded-row">
              <span v-if="excludedTools.length" class="ltr-value">{{ excludedTools.join(', ') }}</span>
              <span v-else class="null-value">None</span>
            </div>
          </article>

          <article class="report-card">
            <span class="report-label">POS conflicts</span>
            <div class="conflict-mini-grid">
              <div v-for="(conflict, index) in posConflicts" :key="index" class="conflict-card">
                <strong class="arabic-value">{{ conflict.word || conflict.token || `#${index + 1}` }}</strong>
                <span class="conflict-badge">{{ conflict.feature || 'POS' }}</span>
                <span class="conflict-text">{{ conflictText(conflict) }}</span>
              </div>
              <span v-if="!posConflicts.length" class="null-value">No POS conflicts were reported.</span>
            </div>
          </article>

          <article class="report-card">
            <span class="report-label">Raw note</span>
            <p class="metrics-note">{{ rawNote || 'The backend did not return an additional note for this run.' }}</p>
          </article>
        </div>
      </section>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import axios from 'axios'
import ToolBadge from '@/components/badges/ToolBadge.vue'
import ScientificChart from '@/components/charts/ScientificChart.vue'
import { API_BASE_URL } from '@/api/nlpApi'
import { recordAnalysis } from '@/utils/analysisHistory'

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const evalResult = ref(null)
const activeTools = ref([])
const excludedTools = ref([])
const posConflicts = ref([])
const rawNote = ref('')
const requestDuration = ref(0)

const EXAMPLE_SENTENCES = [
  '\u0642\u0631\u0623 \u0627\u0644\u0637\u0627\u0644\u0628 \u0627\u0644\u0643\u062a\u0628 \u0641\u064a \u0627\u0644\u0645\u0643\u062a\u0628\u0629',
  '\u0648\u062c\u062f\u062a \u0627\u0644\u0645\u0639\u0644\u0645\u0629 \u0637\u0627\u0644\u0628\u0629 \u0645\u062c\u062a\u0647\u062f\u0629 \u0641\u064a \u0627\u0644\u0641\u0635\u0644',
]

const overallRadar = computed(() => ({
  labels: ['POS', 'Lemma match', 'Coverage', 'Exact lemma'],
  datasets: [
    {
      label: 'Score %',
      data: evalResult.value
        ? [
            toPercent(evalResult.value.pos_agreement),
            toPercent(evalResult.value.lemma_normalized_match),
            toPercent(evalResult.value.segmentation_coverage),
            toPercent(evalResult.value.lemma_exact_match),
          ]
        : [0, 0, 0, 0],
      borderColor: '#14B8A6',
      backgroundColor: 'rgba(20, 184, 166, 0.16)',
    },
  ],
}))

const performanceBar = computed(() => ({
  labels: ['POS', 'Lemma', 'Coverage', 'Exact'],
  datasets: [
    {
      label: 'Percentage',
      data: evalResult.value
        ? [
            toPercent(evalResult.value.pos_agreement),
            toPercent(evalResult.value.lemma_normalized_match),
            toPercent(evalResult.value.segmentation_coverage),
            toPercent(evalResult.value.lemma_exact_match),
          ]
        : [0, 0, 0, 0],
      backgroundColor: ['#4F46E5', '#14B8A6', '#D97706', '#7C3AED'],
    },
  ],
}))

const runtimeBar = computed(() => ({
  labels: ['Request ms', 'Tokens', 'Active tools'],
  datasets: [
    {
      label: 'Current run',
      data: [
        requestDuration.value,
        inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0,
        activeTools.value.length,
      ],
      backgroundColor: ['#4F46E5', '#14B8A6', '#D97706'],
    },
  ],
}))

async function runEvaluation() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  evalResult.value = null
  activeTools.value = []
  excludedTools.value = []
  posConflicts.value = []
  rawNote.value = ''
  requestDuration.value = 0

  try {
    const started = performance.now()
    const { data } = await axios.get(`${API_BASE_URL}/evaluate`, {
      params: { text: inputText.value },
    })
    requestDuration.value = Math.round(performance.now() - started)

    const raw = data?.evaluation || data

    evalResult.value = {
      pos_agreement: toScalar(raw.pos_agreement ?? raw.pos_agreement_pct),
      lemma_normalized_match: toScalar(
        raw.lemma_normalized_match ??
          raw.lemma_normalized_match_pct ??
          raw.lemma_match ??
          raw.lemma_match_pct,
      ),
      lemma_exact_match: toScalar(
        raw.lemma_exact_match ??
          raw.lemma_exact_match_pct ??
          raw.lemma_match ??
          raw.lemma_match_pct,
      ),
      segmentation_coverage: toScalar(raw.segmentation_coverage),
    }

    activeTools.value = raw.active_tools || data.active_tools || []
    excludedTools.value = raw.excluded_tools || data.excluded_tools || []
    posConflicts.value = raw.pos_conflicts || raw.all_conflicts || []
    rawNote.value = raw.metrics_note || ''

    recordAnalysis({
      page: 'Evaluate',
      text: inputText.value.trim(),
      summary: `${toPercent(evalResult.value.pos_agreement).toFixed(1)}% POS agreement | ${requestDuration.value} ms`,
    })
  } catch (e) {
    error.value = e?.response?.data?.detail || e?.message || 'Unable to connect to the evaluation service.'
  } finally {
    loading.value = false
  }
}

function runExample(example) {
  inputText.value = example
  runEvaluation()
}

function toScalar(value) {
  if (value == null) return 0
  if (typeof value === 'number') return value > 1 ? value / 100 : value
  if (typeof value === 'string') {
    const n = Number.parseFloat(value.replace('%', ''))
    if (!Number.isFinite(n)) return 0
    return n > 1 ? n / 100 : n
  }
  return 0
}

function toPercent(scalar) {
  return Math.round(toScalar(scalar) * 100)
}

function percentLabel(score, digits = 1) {
  return `${(toScalar(score) * 100).toFixed(digits)}%`
}

function getScoreClass(score) {
  const n = toScalar(score)
  if (n >= 0.85) return 'score-high'
  if (n >= 0.6) return 'score-medium'
  return 'score-low'
}

function conflictText(conflict) {
  const valueA = conflict.tool_a_value || conflict.value_a || conflict.camel_pos || ''
  const valueB = conflict.tool_b_value || conflict.value_b || conflict.stanza_pos || ''
  const toolA = conflict.tool_a || 'CAMeL'
  const toolB = conflict.tool_b || 'Stanza'
  return `${toolA}: ${valueA || '-'} / ${toolB}: ${valueB || '-'}`
}
</script>

<style scoped>
.eval-page {
  display: grid;
  gap: 18px;
}

.eval-hero {
  min-height: 240px;
}

.input-section {
  margin-bottom: 0;
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

.input-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  margin-top: 8px;
}

.arabic-input {
  flex: 1;
  min-height: 74px;
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
  min-width: 156px;
  min-height: 52px;
  padding: 12px 24px;
  border-radius: var(--radius-control);
  color: white;
  background: linear-gradient(135deg, var(--c-text-primary), var(--c-accent));
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
}

.run-btn:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.examples-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.example-chip {
  padding: 5px 12px;
  border: 1px solid var(--c-border);
  border-radius: 999px;
  color: var(--c-text-secondary);
  background: var(--c-page-bg);
  cursor: pointer;
  font-size: 13px;
}

.error-banner {
  padding: 12px 16px;
  border: 1px solid var(--c-conf-low-border);
  border-radius: var(--radius-control);
  color: var(--c-conf-low-text);
  background: var(--c-conf-low-bg);
}

.eval-loading {
  min-height: 160px;
}

.metrics-section {
  display: grid;
  gap: 18px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.kpi-card {
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
  display: grid;
  gap: 6px;
}

.kpi-label {
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.kpi-value {
  font-size: 28px;
  font-weight: 700;
}

.kpi-value.score-high {
  color: #059669;
}

.kpi-value.score-medium {
  color: #D97706;
}

.kpi-value.score-low {
  color: #DC2626;
}

.kpi-note {
  color: var(--c-text-muted);
  font-size: 12px;
}

.analysis-visual-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.report-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.report-card {
  display: grid;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.report-label {
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.methodology-link {
  align-self: center;
  padding: 8px 12px;
  border-radius: 999px;
  color: var(--c-accent-text);
  background: var(--c-accent-light);
  font-size: 13px;
  font-weight: 600;
}

.tool-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.excluded-row {
  color: var(--c-text-secondary);
}

.conflict-mini-grid {
  display: grid;
  gap: 10px;
}

.conflict-card {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  padding: 12px;
  border: 1px solid var(--c-conf-med-border);
  border-radius: var(--radius-control);
  background: var(--c-conf-med-bg);
}

.conflict-text {
  color: var(--c-text-secondary);
  direction: ltr;
  font-size: 13px;
  font-weight: 500;
}

.metrics-note {
  margin: 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

.null-value {
  color: var(--c-text-muted);
  font-size: 13px;
}

@media (max-width: 900px) {
  .kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .analysis-visual-grid,
  .report-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .input-row {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .run-btn {
    width: 100%;
  }

  .kpi-grid {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
