<!-- Version: 8.3.2 -->
<template>
  <div class="page-wrap compare-page page-stack">
    <section class="hero-band compare-hero">
      <div class="hero-content">
        <span class="eyebrow">Comparative NLP dashboard</span>
        <h1 class="hero-title">Token-level evidence, agreement metrics, and conflict diagnostics.</h1>
        <p class="hero-copy">
          Compare CAMeL, Farasa, Stanza, and Qalsadi outputs in one research table.
          Metrics are computed by the backend evaluation endpoint.
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
        placeholder="اكتب النص العربي هنا..."
      ></textarea>

      <div class="run-row">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="compare">
          {{ loading ? 'Running comparison...' : 'Run Comparative Analysis' }}
        </button>
        <button class="btn btn-secondary" :disabled="!results" @click="copyResults">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !results }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !results }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

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

    <template v-if="results && !loading">
      <section v-if="unavailableTools.length" class="panel panel-pad unavailable-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Unavailable Tools</h2>
            <p class="section-subtitle">Safe fallbacks are active; comparison continues with available core evidence.</p>
          </div>
        </div>
        <div class="unavailable-list">
          <span v-for="tool in unavailableTools" :key="tool.tool" class="unavailable-chip">
            {{ tool.tool }}: {{ tool.reason || tool.status }}
          </span>
        </div>
      </section>

      <section class="metrics-grid" aria-label="Evaluation metrics">
        <article v-for="metric in metricCards" :key="metric.key" class="panel panel-pad metric-card">
          <div class="metric-top">
            <span class="metric-icon">{{ metric.icon }}</span>
            <span class="tool-badge" :title="metric.help">Backend metric</span>
          </div>
          <div class="metric-main">
            <div class="ring" :style="{ '--pct': metric.percent }">
              <span>{{ metric.display }}</span>
            </div>
            <div>
              <h3>{{ metric.label }}</h3>
              <p>{{ metric.help }}</p>
            </div>
          </div>
          <div class="progress-track">
            <span :style="{ width: `${metric.percent}%` }"></span>
          </div>
        </article>
      </section>

      <section class="evidence-layout">
        <article class="panel panel-pad token-strip-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Token Visualization</h2>
              <p class="section-subtitle">Each chip shows token, POS agreement, and Qalsadi lemma evidence.</p>
            </div>
          </div>
          <div class="token-strip">
            <button
              v-for="row in rows"
              :key="`chip-${row.index}`"
              type="button"
              :class="['token-chip', row.state]"
              :title="row.reason || row.qalsadi.lemma || row.word"
            >
              <span class="arabic" dir="rtl">{{ row.word }}</span>
              <small>{{ row.camel.pos || row.stanza.pos || 'UNK' }}</small>
            </button>
          </div>
        </article>

        <aside class="panel panel-pad conflict-panel">
          <h2 class="section-title">Conflict Summary</h2>
          <p class="section-subtitle">Rows needing manual review are listed first.</p>
          <div v-if="conflictRows.length" class="conflict-list">
            <div v-for="row in conflictRows" :key="`conflict-${row.index}`" class="conflict-card">
              <strong class="arabic" dir="rtl">{{ row.word }}</strong>
              <span>{{ row.reason }}</span>
            </div>
          </div>
          <div v-else class="clean-state">No CAMeL/Stanza POS conflicts detected.</div>
        </aside>
      </section>

      <section class="panel panel-pad table-panel">
        <div class="section-head">
          <div>
            <h2 class="section-title">Professional NLP Comparison Table</h2>
            <p class="section-subtitle">Aligned by token index with normalized POS labels and visible review states.</p>
          </div>
          <span :class="['pill', agreementPill]">{{ agreementLabel }}</span>
        </div>

        <div class="table-scroll nlp-table">
          <table>
            <thead>
              <tr>
                <th>Token</th>
                <th>CAMeL</th>
                <th>Farasa</th>
                <th>Stanza</th>
                <th>Qalsadi Lemma</th>
                <th>Agreement</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in rows" :key="row.index" :class="['evidence-row', row.state]">
                <td>
                  <div class="token-cell">
                    <strong class="arabic" dir="rtl">{{ row.word }}</strong>
                    <span class="mono">#{{ row.index + 1 }}</span>
                  </div>
                </td>
                <td>
                  <div class="tool-cell">
                    <span :class="posClass(row.camel.pos)">{{ value(row.camel.pos) }}</span>
                    <span class="arabic lemma" dir="rtl">{{ value(row.camel.lemma) }}</span>
                    <small>root {{ value(row.camel.root) }}</small>
                  </div>
                </td>
                <td>
                  <div class="segment-cell">
                    <span v-for="part in row.farasa.parts" :key="`${row.index}-${part}`">{{ part }}</span>
                    <small v-if="!row.farasa.parts.length">No segmentation</small>
                  </div>
                </td>
                <td>
                  <div class="tool-cell">
                    <span :class="posClass(row.stanza.pos)">{{ value(row.stanza.pos) }}</span>
                    <span class="arabic lemma" dir="rtl">{{ value(row.stanza.lemma) }}</span>
                    <small>{{ row.stanza.dep }}</small>
                  </div>
                </td>
                <td>
                  <span class="qalsadi-lemma arabic" dir="rtl">{{ value(row.qalsadi.lemma) }}</span>
                </td>
                <td>
                  <div class="agreement-cell">
                    <span :class="['agreement-badge', row.state]">{{ row.label }}</span>
                    <span v-if="row.reason" class="reason-text">{{ row.reason }}</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { analyzeAll, evaluateText, exportUrl } from '../api/nlpApi'

const POS_UNIFIED = {
  ADJECTIVE: 'ADJ',
  ADPOSITION: 'ADP',
  ADVERB: 'ADV',
  CONJUNCTION: 'CCONJ',
  PARTICLE: 'PART',
  PRONOUN: 'PRON',
  PUNCTUATION: 'PUNCT',
}

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const evalData = ref(null)
const copied = ref(false)

const tokenEstimate = computed(() => inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0)

const rows = computed(() => {
  const camel = results.value?.camel?.tokens || []
  const stanza = results.value?.stanza?.tokens || []
  const farasa = results.value?.farasa?.tokens || []
  const qalsadi = results.value?.qalsadi?.tokens || []
  const max = Math.max(camel.length, stanza.length, farasa.length, qalsadi.length)

  return Array.from({ length: max }, (_, index) => {
    const camelToken = camel[index] || {}
    const camelBest = camelToken.analyses?.[0] || {}
    const stanzaToken = stanza[index] || {}
    const farasaToken = farasa[index] || {}
    const qalsadiToken = qalsadi[index] || {}
    const camelPos = normalizePos(camelBest.pos)
    const stanzaPos = normalizePos(stanzaToken.upos)
    const hasBoth = Boolean(camelPos && stanzaPos)
    const agrees = hasBoth && camelPos === stanzaPos
    const state = agrees ? 'agree' : hasBoth ? 'conflict' : 'partial'

    return {
      index,
      state,
      label: state === 'agree' ? 'Agreement' : state === 'partial' ? 'Partial' : 'Conflict',
      reason: state === 'agree' ? '' : `CAMeL: ${value(camelPos)} | Stanza: ${value(stanzaPos)}`,
      word: camelToken.surface || stanzaToken.surface || farasaToken.surface || qalsadiToken.surface || `#${index + 1}`,
      camel: { pos: camelPos, lemma: camelBest.lemma, root: camelBest.root },
      farasa: { parts: Array.isArray(farasaToken.segmentation) ? farasaToken.segmentation : [] },
      stanza: {
        pos: stanzaPos,
        lemma: stanzaToken.lemma,
        dep: stanzaToken.dependency?.deprel ? `${stanzaToken.dependency.deprel} -> ${stanzaToken.dependency.head_text || stanzaToken.dependency.head || 'root'}` : '-',
      },
      qalsadi: { lemma: qalsadiToken.lemma },
    }
  })
})

const conflictRows = computed(() => rows.value.filter((row) => row.state !== 'agree'))
const unavailableTools = computed(() =>
  Object.entries(results.value || {})
    .filter(([key, payload]) => key !== 'input' && payload && payload.status && payload.status !== 'ok')
    .map(([, payload]) => payload),
)

const metricCards = computed(() => {
  const evaluation = evalData.value || {}
  return [
    {
      key: 'pos',
      label: 'POS Agreement',
      icon: 'POS',
      display: evaluation.pos_agreement_pct || '0%',
      percent: parsePercent(evaluation.pos_agreement_pct),
      help: 'Percentage of tokens where CAMeL and Stanza POS agree.',
    },
    {
      key: 'lemma',
      label: 'Lemma Match',
      icon: 'LEM',
      display: evaluation.lemma_match_pct || '0%',
      percent: parsePercent(evaluation.lemma_match_pct),
      help: 'Backend lemma match rate after diacritic normalization.',
    },
    {
      key: 'f1',
      label: 'POS F1',
      icon: 'F1',
      display: formatDecimal(evaluation.pos_f1),
      percent: toPercent(evaluation.pos_f1),
      help: 'Precision/recall harmonic score for POS agreement.',
    },
    {
      key: 'seg',
      label: 'Segmentation Coverage',
      icon: 'SEG',
      display: `${Math.round((evaluation.segmentation_coverage || 0) * 100)}%`,
      percent: toPercent(evaluation.segmentation_coverage),
      help: 'Share of tokens with Farasa segmentation evidence.',
    },
  ]
})

const agreementPct = computed(() => parsePercent(evalData.value?.pos_agreement_pct))
const agreementPill = computed(() => agreementPct.value >= 75 ? 'pill-green' : agreementPct.value >= 50 ? 'pill-amber' : 'pill-red')
const agreementLabel = computed(() => agreementPct.value >= 75 ? 'High agreement' : agreementPct.value >= 50 ? 'Needs review' : 'High conflict')
const jsonExportHref = computed(() => results.value ? exportUrl(inputText.value, 'json') : '#')
const csvExportHref = computed(() => results.value ? exportUrl(inputText.value, 'csv') : '#')

async function compare() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  evalData.value = null
  copied.value = false

  try {
    const [analysis, evaluation] = await Promise.all([
      analyzeAll(inputText.value),
      evaluateText(inputText.value),
    ])
    results.value = analysis
    evalData.value = evaluation.evaluation || {}
  } catch (e) {
    error.value = e.message || 'Failed to connect to the backend.'
  } finally {
    loading.value = false
  }
}

async function copyResults() {
  if (!results.value) return
  const payload = JSON.stringify({ analysis: results.value, evaluation: evalData.value }, null, 2)
  await navigator.clipboard.writeText(payload)
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}

function guardExport(event) {
  if (!results.value) event.preventDefault()
}

function loadSample() {
  inputText.value = 'قرأ الطلاب الكتب في المكتبة'
}

function clear() {
  inputText.value = ''
  results.value = null
  evalData.value = null
  error.value = ''
  copied.value = false
}

function normalizePos(pos) {
  const value = String(pos || '').toUpperCase()
  return POS_UNIFIED[value] || value
}

function posClass(pos) {
  const value = normalizePos(pos)
  if (value === 'VERB') return 'pos-badge pos-verb'
  if (value === 'NOUN') return 'pos-badge pos-noun'
  if (value === 'ADJ') return 'pos-badge pos-adj'
  if (value === 'ADP' || value === 'PART') return 'pos-badge pos-adp'
  return 'pos-badge pos-other'
}

function parsePercent(value) {
  const parsed = Number.parseFloat(String(value || '').replace('%', ''))
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0
}

function toPercent(value) {
  const numeric = Number(value || 0)
  return Math.max(0, Math.min(100, Math.round(numeric * 100)))
}

function formatDecimal(value) {
  return typeof value === 'number' ? value.toFixed(3) : '0.000'
}

function value(item) {
  return item || '-'
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

.unavailable-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.unavailable-chip {
  display: inline-flex;
  max-width: 100%;
  padding: 8px 10px;
  border: 1px solid #fcd34d;
  border-radius: 8px;
  color: #92400e;
  background: #fffbeb;
  font-size: 12px;
  font-weight: 800;
}

.loading-grid,
.metrics-grid {
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

.metric-card {
  display: grid;
  gap: 16px;
}

.metric-top,
.metric-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.metric-icon {
  width: 38px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  color: #075985;
  background: #e0f2fe;
  font-size: 12px;
  font-weight: 950;
}

.ring {
  --pct: 0;
  width: 76px;
  height: 76px;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background:
    radial-gradient(circle at center, white 58%, transparent 60%),
    conic-gradient(var(--blue) calc(var(--pct) * 1%), #e7edf5 0);
}

.ring span {
  font-size: 14px;
  font-weight: 950;
}

.metric-main h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 950;
}

.metric-main p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
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
  transition: width 0.35s ease;
}

.evidence-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 18px;
}

.token-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.token-chip {
  min-width: 96px;
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  cursor: default;
  text-align: center;
}

.token-chip span {
  font-size: 20px;
  font-weight: 900;
}

.token-chip small {
  color: var(--muted);
  font-size: 11px;
  font-weight: 900;
}

.token-chip.agree {
  border-color: rgba(21, 128, 61, 0.28);
  background: #f3fcf6;
}

.token-chip.partial {
  border-color: rgba(180, 83, 9, 0.30);
  background: #fffbeb;
}

.token-chip.conflict {
  border-color: rgba(180, 35, 24, 0.28);
  background: #fff7f7;
}

.conflict-list {
  display: grid;
  gap: 10px;
  margin-top: 16px;
}

.conflict-card {
  display: grid;
  gap: 6px;
  padding: 12px;
  border: 1px solid #fecaca;
  border-radius: 8px;
  background: #fff7f7;
}

.conflict-card strong {
  color: var(--red);
  font-size: 18px;
  font-weight: 900;
}

.conflict-card span,
.clean-state {
  color: var(--muted);
  font-size: 13px;
  font-weight: 750;
}

.clean-state {
  margin-top: 16px;
  padding: 14px;
  border-radius: 8px;
  background: #f3fcf6;
}

.nlp-table {
  max-height: 620px;
}

.evidence-row.agree td {
  box-shadow: inset 4px 0 0 rgba(21, 128, 61, 0.55);
}

.evidence-row.partial td {
  box-shadow: inset 4px 0 0 rgba(180, 83, 9, 0.55);
}

.evidence-row.conflict td {
  box-shadow: inset 4px 0 0 rgba(180, 35, 24, 0.58);
}

.token-cell,
.tool-cell,
.agreement-cell {
  display: grid;
  gap: 7px;
}

.token-cell strong {
  font-size: 21px;
  font-weight: 950;
}

.token-cell .mono,
.tool-cell small {
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}

.lemma {
  color: var(--slate);
  font-size: 16px;
  font-weight: 850;
}

.segment-cell {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.segment-cell span {
  padding: 5px 8px;
  border-radius: 7px;
  background: #f1f5f9;
  color: var(--violet);
  font-size: 12px;
  font-weight: 900;
}

.qalsadi-lemma {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  padding: 5px 10px;
  border-radius: 8px;
  color: var(--navy);
  background: #eef6ff;
  font-size: 17px;
  font-weight: 900;
}

.agreement-badge {
  width: fit-content;
  min-height: 28px;
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 7px;
  font-size: 12px;
  font-weight: 900;
}

.agreement-badge.agree {
  color: #166534;
  background: #dcfce7;
}

.agreement-badge.partial {
  color: #92400e;
  background: #fef3c7;
}

.agreement-badge.conflict {
  color: #991b1b;
  background: #fee2e2;
}

.reason-text {
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}

@media (max-width: 1100px) {
  .loading-grid,
  .metrics-grid,
  .evidence-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .conflict-panel {
    grid-column: 1 / -1;
  }
}

@media (max-width: 720px) {
  .loading-grid,
  .metrics-grid,
  .evidence-layout {
    grid-template-columns: 1fr;
  }

  .metric-main {
    align-items: flex-start;
  }
}
</style>
