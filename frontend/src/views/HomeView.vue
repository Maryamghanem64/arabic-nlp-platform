<template>
  <div class="page-wrap home-page page-stack">
    <section class="hero-band dashboard-hero">
      <div class="hero-content">
        <span class="eyebrow">Dashboard</span>
        <h1 class="hero-title">Comparative Arabic NLP Platform</h1>
        <p class="hero-copy">
          A research-grade workspace for tool health, benchmark runs, comparison, fusion, and evaluation.
          Arabic is reserved for input and linguistic output, while the interface stays in English.
        </p>
        <div class="actions-row">
          <RouterLink class="btn btn-primary" to="/smart">Open Fusion View</RouterLink>
          <RouterLink class="btn btn-secondary" to="/compare">Compare Tools</RouterLink>
          <RouterLink class="btn btn-subtle" to="/evaluate">View Evaluation</RouterLink>
        </div>
      </div>

      <div class="hero-panel">
        <article class="hero-stat hero-stat--accent">
          <strong>{{ dashboardMetrics.activeTools }}</strong>
          <span>Active tools</span>
        </article>
        <article class="hero-stat">
          <strong>{{ dashboardMetrics.tasks }}</strong>
          <span>NLP tasks covered</span>
        </article>
        <article class="hero-stat">
          <strong>{{ dashboardMetrics.healthLabel }}</strong>
          <span>System health</span>
        </article>
      </div>
    </section>

    <section class="dashboard-grid">
      <article class="panel panel-pad summary-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Live Platform Summary</h2>
            <p class="section-subtitle">Values update from backend health checks and a built-in benchmark sentence.</p>
          </div>
          <button class="btn btn-secondary" :disabled="statusLoading || benchmarkLoading" @click="refreshDashboard">
            {{ statusLoading || benchmarkLoading ? 'Refreshing...' : 'Refresh' }}
          </button>
        </div>

        <div v-if="statusError" class="error-state dashboard-error">
          <div>
            <strong>Backend status unavailable</strong>
            <p>{{ statusError.message || 'Could not reach the tool registry endpoint.' }}</p>
          </div>
        </div>

        <div class="kpi-grid">
          <article v-for="metric in metrics" :key="metric.label" class="kpi-card dashboard-kpi">
            <div class="kpi-label">{{ metric.label }}</div>
            <div class="kpi-value" :class="metric.className">{{ metric.value }}</div>
            <div class="kpi-note">{{ metric.note }}</div>
          </article>
        </div>
      </article>

      <aside class="panel panel-pad benchmark-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Benchmark Run</h2>
            <p class="section-subtitle">{{ benchmarkLabel }}</p>
          </div>
        </div>

        <div v-if="benchmarkLoading" class="loading-state benchmark-loading">
          <span class="spinner--dark" aria-hidden="true"></span>
          <p>Running the reference sentence through fusion and evaluation...</p>
        </div>

        <div v-else class="benchmark-stack">
          <article class="metric-strip">
            <div class="metric-strip-head">
              <span class="metric-label">Average agreement</span>
              <strong>{{ benchmark.agreement }}</strong>
            </div>
            <div class="progress-track"><span :style="{ width: benchmark.agreementWidth }"></span></div>
          </article>

          <article class="metric-strip">
            <div class="metric-strip-head">
              <span class="metric-label">Average confidence</span>
              <strong>{{ benchmark.confidence }}</strong>
            </div>
            <div class="progress-track"><span :style="{ width: benchmark.confidenceWidth }"></span></div>
          </article>

          <article class="metric-strip">
            <div class="metric-strip-head">
              <span class="metric-label">Response time</span>
              <strong>{{ benchmark.responseTime }}</strong>
            </div>
            <div class="progress-track"><span :style="{ width: benchmark.responseWidth }"></span></div>
          </article>

          <article class="metric-strip">
            <span class="metric-label">Benchmark status</span>
            <span :class="['pill', benchmark.statusClass]">{{ benchmark.status }}</span>
            <p class="metrics-note">{{ benchmark.note }}</p>
          </article>
        </div>
      </aside>
    </section>

    <section class="dashboard-grid dashboard-grid--secondary">
      <article class="panel panel-pad chart-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Tool Health</h2>
            <p class="section-subtitle">Grouped by linguistic family.</p>
          </div>
        </div>

        <div class="group-bars">
          <div v-for="group in groupHealth" :key="group.key" class="group-bar-card">
            <div class="group-bar-head">
              <span>{{ group.label }}</span>
              <strong>{{ group.active }}/{{ group.total }}</strong>
            </div>
            <div class="progress-track progress-track--group">
              <span :style="{ width: `${group.ratio}%`, background: group.gradient }"></span>
            </div>
            <small>{{ group.summary }}</small>
          </div>
        </div>

        <svg class="mini-chart" viewBox="0 0 620 220" role="img" aria-label="Tool readiness chart">
          <defs>
            <linearGradient id="readinessGradient" x1="0%" x2="100%" y1="0%" y2="0%">
              <stop offset="0%" stop-color="#4F46E5" />
              <stop offset="100%" stop-color="#14B8A6" />
            </linearGradient>
          </defs>
          <rect x="24" y="24" width="572" height="172" rx="22" fill="rgba(255,255,255,0.78)" stroke="rgba(148,163,184,0.22)" />
          <g v-for="(bar, index) in readinessBars" :key="bar.label">
            <rect :x="70 + index * 165" :y="160 - bar.height" width="86" :height="bar.height" rx="14" fill="url(#readinessGradient)" />
            <text :x="113 + index * 165" y="184" text-anchor="middle" class="chart-label">{{ bar.label }}</text>
            <text :x="113 + index * 165" :y="150 - bar.height" text-anchor="middle" class="chart-value">{{ bar.value }}</text>
          </g>
        </svg>
      </article>

      <article class="panel panel-pad capability-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Platform Capabilities</h2>
            <p class="section-subtitle">The pages are shaped around actual research workflows.</p>
          </div>
        </div>

        <div class="capability-grid">
          <article v-for="item in capabilities" :key="item.label" class="capability-card">
            <span class="capability-label">{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
            <p>{{ item.note }}</p>
          </article>
        </div>
      </article>
    </section>

    <section class="analysis-visual-grid">
      <ScientificChart
        type="doughnut"
        title="Tool Availability Mix"
        subtitle="Online versus offline or degraded tools."
        badge="Live"
        :labels="toolAvailabilityChart.labels"
        :datasets="toolAvailabilityChart.datasets"
        :height="260"
        aria-label="Tool availability doughnut chart"
        empty-title="Waiting for tool status"
        empty-text="The chart will populate as soon as the backend status endpoint responds."
      />

      <ScientificChart
        type="bar"
        title="Benchmark Snapshot"
        subtitle="Agreement, confidence, and latency from the active reference run."
        badge="Benchmark"
        :labels="benchmarkChart.labels"
        :datasets="benchmarkChart.datasets"
        :height="260"
        aria-label="Benchmark snapshot bar chart"
        empty-title="Waiting for benchmark data"
        empty-text="Run the benchmark once to populate the research summary."
      />

      <ScientificChart
        type="radar"
        title="Group Coverage"
        subtitle="Morphology, syntax, and segmentation coverage."
        badge="Coverage"
        :labels="groupCoverageChart.labels"
        :datasets="groupCoverageChart.datasets"
        :height="280"
        aria-label="Coverage radar chart"
        empty-title="Waiting for coverage data"
        empty-text="Coverage chart appears after the tool registry loads."
      />
    </section>

    <section class="dashboard-grid dashboard-grid--tertiary">
      <article class="panel panel-pad services-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Running Services</h2>
            <p class="section-subtitle">Live tool status from the backend registry.</p>
          </div>
        </div>

        <div class="service-grid">
          <article
            v-for="tool in toolCards"
            :key="tool.key"
            :class="['service-card', { unavailable: tool.status !== 'ok' && tool.status !== 'partial' && tool.status !== 'lazy' }]"
          >
            <div class="service-card-head">
              <span class="tool-code">{{ tool.code }}</span>
              <span :class="['pill', statusPill(tool.status)]">{{ readableStatus(tool.status) }}</span>
            </div>
            <h3>{{ tool.name }}</h3>
            <p>{{ tool.description }}</p>
            <small v-if="tool.reason" class="tool-reason">{{ tool.reason }}</small>
          </article>
        </div>
      </article>

      <article class="panel panel-pad corpus-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Available Corpora</h2>
            <p class="section-subtitle">Files and benchmark sets already present in the project workspace.</p>
          </div>
        </div>

        <div class="corpus-list">
          <article v-for="corpus in corpora" :key="corpus.name" class="corpus-card">
            <strong>{{ corpus.name }}</strong>
            <p>{{ corpus.description }}</p>
          </article>
        </div>
      </article>
    </section>

    <section class="dashboard-grid dashboard-grid--secondary">
      <article class="panel panel-pad recent-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Recent Analyses</h2>
            <p class="section-subtitle">Stored locally from previous frontend analysis runs.</p>
          </div>
        </div>

        <div v-if="recentAnalyses.length" class="recent-list">
          <article v-for="entry in recentAnalyses" :key="entry.id" class="recent-card">
            <div class="recent-card-head">
              <strong>{{ entry.page }}</strong>
              <span class="recent-time">{{ formatRelativeTime(entry.at) }}</span>
            </div>
            <p>{{ entry.text }}</p>
            <div v-if="entry.summary" class="recent-summary">{{ entry.summary }}</div>
          </article>
        </div>
        <div v-else class="empty-state recent-empty">
          <div>
            <strong>No recent analyses yet</strong>
            <p>Run Analyze, Compare, or Fusion once and the latest experiments will appear here.</p>
          </div>
        </div>
      </article>

      <article class="panel panel-pad quick-actions-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Quick Actions</h2>
            <p class="section-subtitle">Fast navigation for defense-day workflows.</p>
          </div>
        </div>

        <div class="quick-actions-grid">
          <RouterLink to="/analyze" class="quick-action-card">
            <strong>Single-tool analysis</strong>
            <span>Inspect one analyzer in depth.</span>
          </RouterLink>
          <RouterLink to="/compare" class="quick-action-card">
            <strong>Comparison matrix</strong>
            <span>Review disagreements and agreement metrics.</span>
          </RouterLink>
          <RouterLink to="/smart" class="quick-action-card">
            <strong>Evidence fusion</strong>
            <span>See why each value won the vote.</span>
          </RouterLink>
          <RouterLink to="/evaluate" class="quick-action-card">
            <strong>Scientific report</strong>
            <span>Check benchmark and coverage summaries.</span>
          </RouterLink>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { evaluateText, fusionText } from '@/api/nlpApi'
import { TOOL_CONFIG, TOOL_KEYS } from '@/config/tools'
import { useToolStatus } from '@/composables/useToolStatus'
import { TOOL_GROUPS } from '@/constants/designTokens'
import ScientificChart from '@/components/ScientificChart.vue'
import { readAnalysisHistory } from '@/utils/analysisHistory'

const { toolStatuses, activeTools, loading: statusLoading, error: statusError, refresh } = useToolStatus()

const benchmarkLoading = ref(true)
const benchmarkLabel = ref('Running a benchmark sentence to seed the dashboard...')
const benchmark = ref({
  agreement: '0%',
  agreementWidth: '0%',
  confidence: '0%',
  confidenceWidth: '0%',
  responseTime: '0 ms',
  responseWidth: '0%',
  status: 'Pending',
  statusClass: 'pill-gray',
  note: 'Waiting for the first benchmark run.',
})
const benchmarkMetrics = ref({
  agreement: 0,
  confidence: 0,
  responseTimeMs: 0,
  ok: false,
})

const benchmarkSentence = 'قرأت الطالبة الكتاب في المكتبة'

const toolCards = computed(() =>
  TOOL_KEYS.map((key) => ({
    key,
    code: key.slice(0, 2).toUpperCase(),
    name: TOOL_CONFIG[key].label,
    description: describeTool(key),
    status: toolStatuses.value[key]?.status || 'unknown',
    reason: toolStatuses.value[key]?.reason || toolStatuses.value[key]?.error || '',
  })),
)

const activeToolCount = computed(() => activeTools.value.length)
const totalTools = computed(() => TOOL_KEYS.length)
const supportedTasks = computed(() => {
  const tasks = new Set()
  TOOL_KEYS.forEach((key) => {
    ;(TOOL_CONFIG[key].features || []).forEach((feature) => tasks.add(feature))
  })
  return tasks.size
})

const groupHealth = computed(() =>
  Object.entries(TOOL_GROUPS).map(([key, tools]) => {
    const active = tools.filter((tool) => ['ok', 'partial', 'lazy'].includes(toolStatuses.value[tool]?.status)).length
    const total = tools.length
    const ratio = total ? Math.round((active / total) * 100) : 0
    const label = key === 'morphology' ? 'Morphology' : key === 'syntax' ? 'Syntax' : 'Segmentation'
    const gradient =
      key === 'morphology'
        ? 'linear-gradient(90deg, #7C3AED, #4F46E5)'
        : key === 'syntax'
          ? 'linear-gradient(90deg, #059669, #14B8A6)'
          : 'linear-gradient(90deg, #D97706, #F59E0B)'
    return {
      key,
      label,
      active,
      total,
      ratio,
      gradient,
      summary: `${active} of ${total} tools are currently available.`,
    }
  }),
)

const readinessBars = computed(() => [
  { label: 'Tools', value: `${activeToolCount.value}/${totalTools.value}`, height: Math.max(30, Math.round((activeToolCount.value / Math.max(totalTools.value, 1)) * 120)) },
  { label: 'Tasks', value: `${supportedTasks.value}`, height: Math.max(30, Math.round(Math.min(1, supportedTasks.value / 12) * 120)) },
  { label: 'Health', value: benchmarkMetrics.value.ok ? 'OK' : 'Booting', height: Math.max(30, Math.round((benchmarkMetrics.value.ok ? 1 : 0.4) * 120)) },
])

const metrics = computed(() => [
  {
    label: 'Active tools',
    value: String(activeToolCount.value),
    note: `${activeToolCount.value} online, ${totalTools.value - activeToolCount.value} offline or partial.`,
    className: 'score-high',
  },
  {
    label: 'NLP tasks',
    value: String(supportedTasks.value),
    note: 'Unique tasks exposed across the integrated analyzers.',
    className: 'score-medium',
  },
  {
    label: 'Average agreement',
    value: benchmark.value.agreement,
    note: 'Derived from the built-in benchmark sentence.',
    className: scoreClass(benchmarkMetrics.value.agreement),
  },
  {
    label: 'Average confidence',
    value: benchmark.value.confidence,
    note: 'Fusion confidence from the same benchmark sentence.',
    className: scoreClass(benchmarkMetrics.value.confidence),
  },
  {
    label: 'Response time',
    value: benchmark.value.responseTime,
    note: 'Average of fusion and evaluation calls.',
    className: 'score-medium',
  },
  {
    label: 'System health',
    value: benchmark.value.status,
    note: 'Combines backend availability and benchmark execution.',
    className: benchmark.value.statusClass,
  },
])

const capabilities = computed(() => [
  { label: 'Running services', value: `${activeToolCount.value}/${totalTools.value}`, note: 'The dashboard reflects live startup detection.' },
  { label: 'Supported tasks', value: `${supportedTasks.value}`, note: 'Morphology, segmentation, syntax, and lexical evidence.' },
  { label: 'Benchmark status', value: benchmark.value.status, note: 'The dashboard executes one reference sentence automatically.' },
  { label: 'Evaluation summary', value: benchmark.value.agreement, note: 'Agreement and confidence are pulled from backend endpoints.' },
  { label: 'Average latency', value: benchmark.value.responseTime, note: 'Useful for presenting deployment readiness to reviewers.' },
  { label: 'Corpus sets', value: String(corpora.value.length), note: 'Workspace datasets and exported experiment files.' },
])

const corpora = ref([
  { name: 'evaluate_dataset.json', description: 'Evaluation samples used by the project benchmarking workflow.' },
  { name: 'export_dataset.json', description: 'Export-ready structured data for comparison and reporting.' },
  { name: 'benchmark_progress.jsonl', description: 'Incremental benchmark progress log for reproducibility.' },
])

const recentAnalyses = ref(readAnalysisHistory())

async function refreshDashboard() {
  benchmarkLoading.value = true
  benchmarkLabel.value = 'Refreshing benchmark metrics...'

  try {
    await refresh()
    await runBenchmark()
  } finally {
    benchmarkLoading.value = false
    benchmarkLabel.value = benchmarkMetrics.value.ok
      ? 'Live benchmark data captured from the backend.'
      : 'Benchmark data is unavailable, but the dashboard remains usable.'
  }
}

async function runBenchmark() {
  const started = performance.now()
  try {
    const [evaluationResult, fusionResult] = await Promise.all([evaluateText(benchmarkSentence), fusionText(benchmarkSentence)])
    const finished = performance.now()
    const evaluation = evaluationResult?.evaluation || evaluationResult || {}
    const fusionRows = Array.isArray(fusionResult?.fusion) ? fusionResult.fusion : Array.isArray(fusionResult) ? fusionResult : []
    const confidenceScores = fusionRows
      .map((row) => Number(row?.final?.confidence_score ?? row?.confidence_score ?? row?.confidence?.score))
      .filter((value) => Number.isFinite(value))

    const agreementScore = average([
      parsePercent(evaluation.pos_agreement_pct),
      parsePercent(evaluation.lemma_match_pct),
      parsePercent(evaluation.segmentation_coverage),
    ])
    const confidenceScore = confidenceScores.length ? average(confidenceScores) : 0

    benchmarkMetrics.value = {
      agreement: agreementScore,
      confidence: confidenceScore,
      responseTimeMs: Math.round(finished - started),
      ok: true,
    }

    benchmark.value = {
      agreement: `${Math.round(agreementScore * 100)}%`,
      agreementWidth: `${Math.round(agreementScore * 100)}%`,
      confidence: `${Math.round(confidenceScore * 100)}%`,
      confidenceWidth: `${Math.round(confidenceScore * 100)}%`,
      responseTime: `${Math.round(finished - started)} ms`,
      responseWidth: `${Math.max(18, 100 - Math.min(95, Math.round((finished - started) / 12)))}%`,
      status: 'Ready',
      statusClass: 'pill-green',
      note: `Evaluation and fusion completed in ${Math.round(finished - started)} ms for the reference sentence.`,
    }
  } catch (error) {
    const finished = performance.now()
    benchmarkMetrics.value = {
      agreement: 0,
      confidence: 0,
      responseTimeMs: Math.round(finished - started),
      ok: false,
    }
    benchmark.value = {
      agreement: '0%',
      agreementWidth: '0%',
      confidence: '0%',
      confidenceWidth: '0%',
      responseTime: `${Math.round(finished - started)} ms`,
      responseWidth: '22%',
      status: 'Degraded',
      statusClass: 'pill-amber',
      note: error?.message || 'The benchmark request could not complete on this machine.',
    }
  }
}

function scoreClass(value) {
  if (value >= 0.85) return 'score-high'
  if (value >= 0.6) return 'score-medium'
  return 'score-low'
}

function readableStatus(status) {
  if (status === 'ok') return 'Online'
  if (status === 'partial') return 'Partial'
  if (status === 'lazy') return 'On demand'
  if (status === 'future_work') return 'Planned'
  if (status === 'missing_dependency') return 'Missing dependency'
  if (status === 'missing_model') return 'Missing model'
  if (status === 'missing_java') return 'Missing Java'
  if (status === 'unavailable') return 'Unavailable'
  return 'Offline'
}

function statusPill(status) {
  if (status === 'ok') return 'pill-green'
  if (status === 'partial' || status === 'lazy') return 'pill-amber'
  if (status === 'future_work' || String(status).startsWith('missing') || status === 'unavailable') return 'pill-gray'
  return 'pill-red'
}

function describeTool(key) {
  const descriptions = {
    camel: 'Morphology, lemma, root, and lexical confidence evidence.',
    farasa: 'Segmentation and clitic-aware token handling.',
    stanza: 'POS, lemma, case, and dependency parsing.',
    qalsadi: 'Lightweight rule-based lemma and stem signals.',
    arabert: 'Transformer-based contextual analysis experiments.',
    alkhalil: 'Rule-based morphology and root extraction.',
    udpipe: 'Universal Dependencies syntax and lemma output.',
    madamira: 'Classical Arabic morphological analysis pipeline.',
    sinatools: 'Experimental lemmatization and word-sense services.',
  }
  return descriptions[key] || 'Integrated NLP service.'
}

function parsePercent(value) {
  if (typeof value === 'number') return value > 1 ? value / 100 : value
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value.replace('%', ''))
    return Number.isFinite(parsed) ? parsed / 100 : 0
  }
  return 0
}

function average(values) {
  const items = values.filter((value) => Number.isFinite(value))
  if (!items.length) return 0
  return items.reduce((sum, value) => sum + value, 0) / items.length
}

async function refreshStatusOnly() {
  try {
    await refresh()
  } catch {
    // The dashboard renders a safe fallback state when the backend is not reachable.
  }
}

async function initializeDashboard() {
  await refreshStatusOnly()
  benchmarkLoading.value = true
  await runBenchmark()
  benchmarkLoading.value = false
  benchmarkLabel.value = benchmarkMetrics.value.ok
    ? 'Live benchmark data captured from the backend.'
    : 'Benchmark data is unavailable, but the dashboard remains usable.'
}

function refreshRecentAnalyses() {
  recentAnalyses.value = readAnalysisHistory()
}

const handleHistoryUpdate = () => refreshRecentAnalyses()

const dashboardMetrics = computed(() => ({
  activeTools: activeToolCount.value,
  tasks: supportedTasks.value,
  healthLabel: benchmarkMetrics.value.ok ? 'Healthy' : 'Degraded',
}))

const toolAvailabilityChart = computed(() => {
  const online = activeToolCount.value
  const offline = Math.max(0, totalTools.value - online)
  return {
    labels: ['Online', 'Offline'],
    datasets: [
      {
        label: 'Tools',
        data: [online, offline],
        backgroundColor: ['#14B8A6', '#CBD5E1'],
      },
    ],
  }
})

const benchmarkChart = computed(() => ({
  labels: ['Agreement', 'Confidence', 'Latency'],
  datasets: [
    {
      label: 'Benchmark',
      data: [
        Math.round(benchmarkMetrics.value.agreement * 100),
        Math.round(benchmarkMetrics.value.confidence * 100),
        Math.max(0, 100 - Math.min(100, Math.round(benchmarkMetrics.value.responseTimeMs / 12))),
      ],
      backgroundColor: ['#4F46E5', '#14B8A6', '#D97706'],
    },
  ],
}))

const groupCoverageChart = computed(() => ({
  labels: Object.values(TOOL_GROUPS).map((tools, index) => (index === 0 ? 'Morphology' : index === 1 ? 'Syntax' : 'Segmentation')),
  datasets: [
    {
      label: 'Coverage',
      data: groupHealth.value.map((group) => group.ratio),
      backgroundColor: ['#7C3AED66', '#05966966', '#D9770666'],
      borderColor: ['#7C3AED', '#059669', '#D97706'],
      fill: true,
    },
  ],
}))

function formatRelativeTime(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const minutes = Math.max(1, Math.round(diff / 60000))
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.max(1, Math.round(minutes / 60))
  if (hours < 24) return `${hours}h ago`
  const days = Math.max(1, Math.round(hours / 24))
  return `${days}d ago`
}

onMounted(initializeDashboard)

onMounted(() => {
  window.addEventListener('analysis-history-updated', handleHistoryUpdate)
})

onUnmounted(() => {
  window.removeEventListener('analysis-history-updated', handleHistoryUpdate)
})
</script>

<style scoped>
.dashboard-hero {
  grid-template-columns: minmax(0, 1fr) 300px;
  align-items: stretch;
}

.hero-panel {
  display: grid;
  gap: 12px;
}

.hero-stat {
  padding: 18px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.08);
}

.hero-stat--accent {
  background: linear-gradient(135deg, rgba(224, 231, 255, 0.18), rgba(20, 184, 166, 0.16));
}

.hero-stat strong {
  display: block;
  font-size: 32px;
  line-height: 1;
  font-weight: 700;
}

.hero-stat span {
  display: block;
  margin-top: 6px;
  color: rgba(255, 255, 255, 0.78);
  font-size: 13px;
  font-weight: 500;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.8fr);
  gap: 18px;
}

.dashboard-grid--secondary,
.dashboard-grid--tertiary {
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
}

.analysis-visual-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.summary-panel,
.benchmark-panel,
.chart-panel,
.capability-panel,
.services-panel,
.corpus-panel {
  min-height: 100%;
}

.dashboard-kpi {
  min-height: 146px;
}

.benchmark-loading {
  min-height: 300px;
}

.benchmark-stack {
  display: grid;
  gap: 12px;
}

.metrics-note {
  margin: 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

.group-bars {
  display: grid;
  gap: 12px;
  margin-bottom: 18px;
}

.group-bar-card {
  display: grid;
  gap: 8px;
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.group-bar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--c-text-primary);
  font-weight: 600;
}

.group-bar-card small {
  color: var(--c-text-secondary);
}

.progress-track--group {
  height: 10px;
}

.mini-chart {
  width: 100%;
  height: auto;
}

.chart-label {
  fill: var(--c-text-secondary);
  font-size: 12px;
  font-weight: 600;
}

.chart-value {
  fill: var(--c-text-primary);
  font-size: 14px;
  font-weight: 700;
}

.capability-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.capability-card,
.corpus-card {
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.capability-label {
  display: block;
  margin-bottom: 8px;
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.capability-card strong,
.corpus-card strong {
  display: block;
  color: var(--c-text-primary);
  font-size: 18px;
  font-weight: 700;
}

.capability-card p,
.corpus-card p {
  margin: 6px 0 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

.service-grid,
.corpus-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.service-card {
  display: grid;
  gap: 10px;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 16px;
  background: var(--c-surface);
}

.service-card.unavailable {
  background: var(--c-page-bg);
  opacity: 0.9;
}

.service-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.tool-code {
  width: 38px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 12px;
  font-weight: 700;
}

.service-card h3 {
  margin: 0;
  font-size: 17px;
  font-weight: 700;
}

.service-card p {
  margin: 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

.tool-reason {
  color: var(--c-segment-text);
  line-height: 1.45;
}

.dashboard-error {
  margin-bottom: 14px;
}

.recent-list {
  display: grid;
  gap: 10px;
}

.recent-card {
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.recent-card-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

.recent-card p {
  margin: 8px 0 0;
  color: var(--c-text-secondary);
}

.recent-summary {
  margin-top: 10px;
  color: var(--c-text-primary);
  font-size: 13px;
  font-weight: 600;
}

.recent-empty {
  min-height: 180px;
}

.quick-actions-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.quick-action-card {
  display: grid;
  gap: 8px;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
  text-decoration: none;
}

.quick-action-card strong {
  color: var(--c-text-primary);
  font-size: 15px;
  font-weight: 700;
}

.quick-action-card span {
  color: var(--c-text-secondary);
  line-height: 1.5;
}

@media (max-width: 1100px) {
  .dashboard-hero,
  .dashboard-grid,
  .dashboard-grid--secondary,
  .dashboard-grid--tertiary,
  .analysis-visual-grid,
  .service-grid,
  .corpus-list,
  .capability-grid,
  .quick-actions-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .hero-panel {
    grid-template-columns: 1fr;
  }
}
</style>
