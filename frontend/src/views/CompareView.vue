<!-- Version: 8.3.1 -->
<template>
  <div class="page-wrap compare-page">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Comparison view</span>
      <h1 class="hero-title">Compare Tool Decisions</h1>
      <p class="hero-copy">
        Review token-level POS differences from CAMeL and Stanza, Farasa segmentation,
        and Qalsadi lemmas using the backend evaluation endpoint for metrics.
      </p>
    </section>

    <section class="panel panel-pad input-panel">
      <div class="input-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
          <p class="section-subtitle">Enter Arabic text to compare all tools.</p>
        </div>

        <label class="toggle">
          <input type="checkbox" v-model="showDetails" />
          <span>Show lemmas and roots</span>
        </label>
      </div>

      <textarea
        v-model="inputText"
        class="textarea arabic"
        dir="rtl"
        placeholder="اكتب النص العربي هنا..."
      ></textarea>

      <div class="actions-row form-actions">
        <button
          class="btn btn-primary"
          :disabled="loading || !inputText.trim()"
          @click="compare"
        >
          {{ loading ? 'Comparing...' : 'Compare Tools' }}
        </button>
        <button class="btn btn-secondary" @click="loadSample">Sample</button>
        <button class="btn btn-secondary" @click="clear">Clear</button>
      </div>
    </section>

    <div v-if="loading" class="loading-state">Comparing token outputs...</div>
    <div v-if="error" class="error-state">{{ error }}</div>

    <section v-if="results && !loading" class="comparison-results">
      <div class="metrics-grid">
        <article class="panel panel-pad metric" :class="metricColor(metrics.posAgreementValue)">
          <strong>{{ metrics.posAgreement }}</strong>
          <span>POS agreement</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.lemmaAgreement }}</strong>
          <span>Lemma match</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.posF1 }}</strong>
          <span>POS F1</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.segmentationCoverage }}</strong>
          <span>Segmentation coverage</span>
        </article>
      </div>

      <article class="panel panel-pad">
        <div class="result-head">
          <div>
            <h2 class="section-title">Token Comparison</h2>
            <p class="section-subtitle">Rows are aligned by token index from the combined result.</p>
          </div>
        </div>

        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>CAMeL POS</th>
                <th>Stanza POS</th>
                <th>Qalsadi Lemma</th>
                <th>Farasa Segments</th>
                <th>Agreement</th>
              </tr>
            </thead>

            <tbody>
              <template v-for="row in rows" :key="row.index">
                <tr :class="rowClass(row)">
                  <td class="arabic word-cell" dir="rtl">{{ row.word }}</td>
                  <td><span :class="posClass(row.camel.pos)">{{ value(row.camel.pos) }}</span></td>
                  <td><span :class="posClass(row.stanza.pos)">{{ value(row.stanza.pos) }}</span></td>
                  <td class="arabic qalsadi-cell" dir="rtl">{{ value(row.qalsadi.lemma) }}</td>
                  <td class="segments">{{ row.farasa.segments || '-' }}</td>
                  <td>
                    <div class="agree-cell">
                      <span :class="['pill', row.agrees ? 'pill-green' : 'pill-red']">
                        {{ row.agrees ? 'Agree' : 'Review' }}
                      </span>
                      <span v-if="!row.agrees && row.reason" class="reason-text">
                        {{ row.reason }}
                      </span>
                    </div>
                  </td>
                </tr>

                <tr v-if="showDetails" class="detail-row">
                  <td></td>
                  <td>
                    <strong>Lemma:</strong> {{ value(row.camel.lemma) }}<br />
                    <strong>Root:</strong> {{ value(row.camel.root) }}
                  </td>
                  <td>
                    <strong>Lemma:</strong>
                    <span class="arabic" dir="rtl">{{ value(row.stanza.lemma) }}</span>
                  </td>
                  <td>
                    <strong>Lemma:</strong>
                    <span class="arabic" dir="rtl">{{ value(row.qalsadi.lemma) }}</span>
                  </td>
                  <td colspan="2">
                    <strong>Farasa:</strong> {{ row.farasa.segments || '-' }}
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>

        <div v-if="summary" class="summary-box">
          <div class="sum-header">Analysis Summary</div>
          <div class="sum-grid">
            <div class="sum-item">
              <span class="sum-k">Total Words</span>
              <span class="sum-v">{{ summary.total }}</span>
            </div>
            <div class="sum-item">
              <span class="sum-k">Agree</span>
              <span class="sum-v green">{{ summary.agree }}</span>
            </div>
            <div class="sum-item">
              <span class="sum-k">Conflict</span>
              <span class="sum-v red">{{ summary.conflict }}</span>
            </div>
            <div class="sum-item">
              <span class="sum-k">Confidence</span>
              <span class="sum-v" :class="summary.confClass">{{ summary.confidence }}</span>
            </div>
          </div>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
// Version: 8.3.1
import { computed, ref } from 'vue'
import { analyzeAll, evaluateText } from '../api/nlpApi'

const POS_UNIFIED = {
  ADJECTIVE: 'ADJ',
  ADPOSITION: 'ADP',
  ADVERB: 'ADV',
  CONJUNCTION: 'CCONJ',
  PARTICLE: 'PART',
  PRONOUN: 'PRON',
  PUNCTUATION: 'PUNCT',
}

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const evalData = ref(null)
const showDetails = ref(false)

async function compare() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  evalData.value = null

  try {
    const [analysis, evaluation] = await Promise.all([
      analyzeAll(inputText.value),
      evaluateText(inputText.value),
    ])
    results.value = analysis
    evalData.value = evaluation.evaluation || null
  } catch (e) {
    console.error(e)
    error.value = 'Failed to connect to the backend. Start the FastAPI server and try again.'
  } finally {
    loading.value = false
  }
}

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
    const agrees = posAgrees(camelPos, stanzaPos)

    return {
      index,
      word: camelToken.surface || stanzaToken.surface || farasaToken.surface || qalsadiToken.surface || `#${index + 1}`,
      camel: { pos: camelPos, lemma: camelBest.lemma, root: camelBest.root },
      stanza: { pos: stanzaPos, lemma: stanzaToken.lemma },
      farasa: { segments: Array.isArray(farasaToken.segmentation) ? farasaToken.segmentation.join(' + ') : '' },
      qalsadi: { lemma: qalsadiToken.lemma },
      agrees,
      reason: agrees ? '' : `CAMeL: ${value(camelPos)} | Stanza: ${value(stanzaPos)}`,
    }
  })
})

const metrics = computed(() => {
  const evaluation = evalData.value || {}
  const posAgreement = evaluation.pos_agreement_pct || '0%'
  return {
    posAgreement,
    posAgreementValue: parsePercent(posAgreement),
    lemmaAgreement: evaluation.lemma_match_pct || '0%',
    posF1: typeof evaluation.pos_f1 === 'number' ? evaluation.pos_f1.toFixed(3) : '0.000',
    segmentationCoverage: typeof evaluation.segmentation_coverage === 'number'
      ? `${Math.round(evaluation.segmentation_coverage * 100)}%`
      : '0%',
  }
})

const summary = computed(() => {
  const total = rows.value.length
  const agree = rows.value.filter((row) => row.agrees).length
  const conflict = total - agree
  const pct = total ? Math.round((agree / total) * 100) : 0

  return {
    total,
    agree,
    conflict,
    confidence: `${pct}%`,
    confClass: pct >= 75 ? 'green' : pct >= 50 ? 'amber' : 'red',
  }
})

function normalizePos(pos) {
  const value = String(pos || '').toUpperCase()
  return POS_UNIFIED[value] || value
}

function posAgrees(camelPos, stanzaPos) {
  if (!camelPos || !stanzaPos) return false
  return camelPos === stanzaPos
}

function parsePercent(value) {
  const parsed = Number.parseFloat(String(value).replace('%', ''))
  return Number.isFinite(parsed) ? parsed : 0
}

function metricColor(value) {
  if (value >= 75) return 'metric-good'
  if (value >= 50) return 'metric-warn'
  return 'metric-bad'
}

function rowClass(row) {
  return row.agrees ? 'row-agree' : 'row-conflict'
}

function posClass(pos) {
  const value = normalizePos(pos)
  if (value === 'VERB') return 'pill pill-blue'
  if (value === 'NOUN') return 'pill pill-green'
  if (value === 'ADJ') return 'pill pill-violet'
  if (value === 'ADP') return 'pill pill-amber'
  return 'pill pill-gray'
}

function value(item) {
  return item || '-'
}

function clear() {
  inputText.value = ''
  results.value = null
  evalData.value = null
  error.value = ''
}

function loadSample() {
  inputText.value = 'قرأ الطلاب الكتب في المكتبة'
}
</script>

<style scoped>
.compact-hero { padding: 34px 38px; }
.compact-hero .hero-title { font-size: 38px; }
.input-panel, .comparison-results { margin-top: 18px; }
.input-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:16px; }
.toggle { display:inline-flex; align-items:center; gap:8px; color:var(--muted,#64748b); font-size:14px; font-weight:600; cursor:pointer; }
.form-actions { margin-top: 16px; }

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 18px;
}

.metric {
  display: grid;
  gap: 6px;
}

.metric strong {
  color: var(--navy);
  font-size: 30px;
  line-height: 1;
}

.metric span {
  color: var(--muted);
  font-size: 13px;
  font-weight: 800;
}

.metric-good { border-color: rgba(21, 128, 61, 0.3); }
.metric-warn { border-color: rgba(180, 83, 9, 0.3); }
.metric-bad { border-color: rgba(180, 35, 24, 0.3); }

.word-cell {
  font-size: 18px;
  font-weight: 900;
}

.qalsadi-cell,
.segments {
  color: var(--violet);
  font-weight: 800;
}

.agree-cell {
  display: grid;
  gap: 6px;
}

.reason-text {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.row-conflict td {
  background: #fffafa;
}

.detail-row td {
  background: #fbfdff;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.7;
}

@media (max-width: 860px) {
  .metrics-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .input-head {
    flex-direction: column;
  }
}
</style>
