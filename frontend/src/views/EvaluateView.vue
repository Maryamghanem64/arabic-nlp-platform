<template>
  <div class="page-wrap eval-page page-stack">
    <section class="hero-band eval-hero">
      <div class="hero-content">
        <span class="eyebrow">Evaluation report</span>
        <h1 class="hero-title">Scientific tool agreement and coverage analysis.</h1>
        <p class="hero-copy">
          Evaluate POS agreement, lemma match, segmentation coverage, and conflict density on one Arabic sentence.
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
          placeholder="Example: قرأ الطالب الكتب في المكتبة"
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
          subtitle="Agreement, lemma, and coverage quality."
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
          subtitle="Accuracy and coverage at a glance."
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
            <h2 class="section-title">Evaluation Summary</h2>
            <p class="section-subtitle">This is the committee-friendly snapshot of the backend evaluation output.</p>
          </div>
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
            <span class="report-label">Benchmark note</span>
            <p class="metrics-note">{{ rawNote || 'The backend did not return an additional note for this run.' }}</p>
          </article>
        </div>
      </section>

      <section class="panel panel-pad conclusions-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Scientific Conclusions</h2>
            <p class="section-subtitle">A compact defense-ready interpretation of the current run.</p>
          </div>
        </div>
        <div class="conclusion-grid">
          <article class="conclusion-card">
            <strong>Agreement quality</strong>
            <p>{{ conclusionText.agreement }}</p>
          </article>
          <article class="conclusion-card">
            <strong>Coverage quality</strong>
            <p>{{ conclusionText.coverage }}</p>
          </article>
          <article class="conclusion-card">
            <strong>Runtime note</strong>
            <p>{{ conclusionText.runtime }}</p>
          </article>
          <article class="conclusion-card">
            <strong>Defense takeaway</strong>
            <p>{{ conclusionText.takeaway }}</p>
          </article>
        </div>
      </section>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import axios from 'axios'
import ToolBadge from '@/components/ToolBadge.vue'
import ScientificChart from '@/components/ScientificChart.vue'
import { API_BASE_URL } from '@/api/nlpApi'
import { recordAnalysis } from '@/utils/analysisHistory'

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const payload = ref(null)
const requestDuration = ref(0)

const EXAMPLE_SENTENCES = [
  'قرأ الطالب الكتب في المكتبة',
  'وجدت المعلمة طالبة مجتهدة في الفصل',
]

const evalResult = computed(() => (payload.value ? normalizeEvaluation(payload.value) : null))
const activeTools = computed(() => payload.value?.active_tools || evalResult.value?.active_tools || [])
const excludedTools = computed(() => evalResult.value?.excluded_tools || payload.value?.meta?.degraded_tools || [])
const posConflicts = computed(() => evalResult.value?.pos_conflicts || [])
const rawNote = computed(() => evalResult.value?.metrics_note || '')
const overallRadar = computed(() => ({
  labels: ['POS', 'Lemma match', 'Coverage', 'Exact lemma'],
  datasets: [
    {
      label: 'Score %',
      data: [
        metricPercent(evalResult.value?.pos_agreement),
        metricPercent(evalResult.value?.lemma_normalized_match),
        metricPercent(evalResult.value?.segmentation_coverage),
        metricPercent(evalResult.value?.lemma_exact_match),
      ],
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
      data: [
        metricPercent(evalResult.value?.pos_agreement),
        metricPercent(evalResult.value?.lemma_normalized_match),
        metricPercent(evalResult.value?.segmentation_coverage),
        metricPercent(evalResult.value?.lemma_exact_match),
      ],
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
const conclusionText = computed(() => {
  const agreement = metricPercent(evalResult.value?.pos_agreement)
  const coverage = metricPercent(evalResult.value?.segmentation_coverage)
  return {
    agreement:
      agreement >= 0.8
        ? 'The current run shows strong agreement across the active analyzers.'
        : agreement >= 0.6
          ? 'The current run is moderately consistent and merits token-level inspection.'
          : 'The current run shows low agreement and should be reviewed carefully.',
    coverage:
      coverage >= 0.8
        ? 'Segmentation coverage is strong enough for comparative analysis.'
        : coverage >= 0.5
          ? 'Coverage is usable but uneven across the sentence.'
          : 'Coverage is limited and may affect downstream interpretation.',
    runtime:
      requestDuration.value
        ? `The last evaluation completed in ${requestDuration.value} ms on the current frontend session.`
        : 'Runtime was not captured for this run.',
    takeaway:
      rawNote.value || 'The evaluation should be paired with compare and fusion views during the defense.',
  }
})

async function runEvaluation() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  payload.value = null
  requestDuration.value = 0

  try {
    const started = performance.now()
    const { data } = await axios.get(`${API_BASE_URL}/evaluate`, { params: { text: inputText.value } })
    requestDuration.value = Math.round(performance.now() - started)
    payload.value = data
    recordAnalysis({
      page: 'Evaluate',
      text: inputText.value.trim(),
      summary: `${percentLabel(normalizeEvaluation(data).pos_agreement, 0)} POS agreement | ${requestDuration.value} ms`,
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

function normalizeEvaluation(rawPayload) {
  const raw = rawPayload.evaluation || rawPayload
  const posAgreement = scoreValue(raw.pos_agreement ?? raw.pos_agreement_pct)
  const lemmaNormalized = scoreValue(raw.lemma_normalized_match ?? raw.lemma_normalized_match_pct ?? raw.lemma_match_pct)

  return {
    ...raw,
    pos_agreement: posAgreement,
    lemma_normalized_match: lemmaNormalized,
    lemma_exact_match: scoreValue(raw.lemma_exact_match ?? raw.lemma_exact_match_pct ?? raw.lemma_match_pct),
    segmentation_coverage: scoreValue(raw.segmentation_coverage),
    active_tools: raw.active_tools || rawPayload.active_tools || [],
    excluded_tools: raw.excluded_tools || rawPayload.meta?.degraded_tools || [],
    pos_conflicts: raw.pos_conflicts || [],
  }
}

function scoreValue(value) {
  if (typeof value === 'number') return value > 1 ? value / 100 : value
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value.replace('%', ''))
    return Number.isFinite(parsed) ? parsed / 100 : 0
  }
  return 0
}

function percentLabel(score, digits = 1) {
  return `${(scoreValue(score) * 100).toFixed(digits)}%`
}

function getScoreClass(score) {
  const normalized = scoreValue(score)
  if (normalized >= 0.85) return 'score-high'
  if (normalized >= 0.6) return 'score-medium'
  return 'score-low'
}

function conflictText(conflict) {
  const valueA = conflict.tool_a_value || conflict.value_a || conflict.toolAValue || ''
  const valueB = conflict.tool_b_value || conflict.value_b || conflict.toolBValue || ''
  const toolA = conflict.tool_a || conflict.toolA || 'tool A'
  const toolB = conflict.tool_b || conflict.toolB || 'tool B'
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

.conclusions-panel {
  margin-top: 0;
}

.conclusion-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.conclusion-card {
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.conclusion-card strong {
  display: block;
  color: var(--c-text-primary);
  font-size: 14px;
  font-weight: 700;
}

.conclusion-card p {
  margin: 6px 0 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

@media (max-width: 760px) {
  .input-row,
  .report-grid,
  .conclusion-grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .run-btn {
    width: 100%;
  }

  .analysis-visual-grid {
    grid-template-columns: 1fr;
  }
}
</style>
