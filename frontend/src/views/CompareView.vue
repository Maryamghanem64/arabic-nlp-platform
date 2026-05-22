<!-- Version: 8.3 -->
<template>
  <div class="page-wrap compare-page">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Comparison view</span>
      <h1 class="hero-title">Compare Tool Decisions</h1>
      <p class="hero-copy">Review token-level differences between CAMeL, Stanza, Farasa, and Qalsadi.</p>
    </section>

    <!-- Input -->
    <section class="panel panel-pad input-panel">
      <div class="input-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
          <p class="section-subtitle">Enter Arabic text to compare all tools.</p>
        </div>

        <label class="toggle">
          <input type="checkbox" v-model="showDetails" />
          <span>Show lemmas and stems</span>
        </label>
      </div>

      <textarea
        v-model="inputText"
        class="textarea arabic"
        dir="rtl"
        placeholder="اكتب النص العربي هنا..."
      ></textarea>

      <div class="actions-row form-actions">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="compare">
          {{ loading ? 'Comparing...' : 'Compare Tools' }}
        </button>
        <button class="btn btn-secondary" @click="loadSample">Sample</button>
        <button class="btn btn-secondary" @click="clear">Clear</button>
      </div>
    </section>

    <div v-if="loading" class="loading-state">Loading token outputs... (may take up to 2 minutes)</div>
    <div v-if="error" class="error-state">{{ error }}</div>

    <!-- Results -->
    <section v-if="results && !loading" class="comparison-results">
      <!-- Metrics Cards -->
      <div class="metrics-grid">
        <article class="panel panel-pad metric" :class="metricColor(metrics.posAgreement)">
          <div class="metric-icon">🎯</div>
          <strong>{{ metrics.posAgreement }}%</strong>
          <span>POS Agreement</span>
        </article>

        <article class="panel panel-pad metric" :class="metricColor(metrics.lemmaMatch)">
          <div class="metric-icon">📖</div>
          <strong>{{ metrics.lemmaMatch }}%</strong>
          <span>Lemma Match</span>
        </article>

        <article class="panel panel-pad metric" :class="metricColor(metrics.f1)">
          <div class="metric-icon">📊</div>
          <strong>{{ metrics.f1 }}%</strong>
          <span>F1 Score</span>
        </article>

        <article class="panel panel-pad metric" :class="metricColor(metrics.farasaCoverage)">
          <div class="metric-icon">✂️</div>
          <strong>{{ metrics.farasaCoverage }}%</strong>
          <span>Segmentation Coverage</span>
        </article>
      </div>

      <!-- Comparison Table -->
      <article class="panel panel-pad">
        <div class="result-head">
          <div>
            <h2 class="section-title">Token Comparison</h2>
            <p class="section-subtitle">CAMeL (Statistical) vs Stanza (Neural) vs Qalsadi (Rule-based)</p>
          </div>
        </div>

        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th class="th-camel">CAMeL POS</th>
                <th class="th-stanza">Stanza POS</th>
                <th class="th-qalsadi">Qalsadi Lemma</th>
                <th class="th-farasa">Farasa Segments</th>
                <th>Agreement</th>
              </tr>
            </thead>

            <tbody>
              <template v-for="row in rows" :key="row.index">
                <tr :class="rowClass(row)">
                  <td class="arabic word-cell" dir="rtl">{{ row.word }}</td>

                  <td>
                    <span :class="posClass(row.camel.pos)">{{ value(row.camel.pos) }}</span>
                  </td>

                  <td>
                    <span :class="posClass(row.stanza.pos)">{{ value(row.stanza.pos) }}</span>
                  </td>

                  <!-- Qalsadi column: lemma only -->
                  <td class="arabic qalsadi-cell" dir="rtl">{{ value(row.qalsadi.lemma) }}</td>

                  <td class="segments">{{ row.farasa.segments || '—' }}</td>

                  <td>
                    <div class="agree-cell">
                      <span :class="['pill', row.agrees ? 'pill-green' : 'pill-red']">
                        {{ row.agrees ? '✅ Agree' : '❌ Review' }}
                      </span>
                      <span v-if="!row.agrees && row.reason" class="reason-text">{{ row.reason }}</span>
                    </div>
                  </td>
                </tr>

                <tr v-if="showDetails" class="detail-row">
                  <td></td>
                  <td>
                    <strong>Lemma:</strong>
                    <span class="arabic" dir="rtl">{{ value(row.camel.lemma) }}</span>
                    <br />
                    <strong>Root:</strong>
                    <span class="arabic" dir="rtl">{{ value(row.camel.root) }}</span>
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
                    <strong>Farasa:</strong> {{ row.farasa.segments || '—' }}
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </article>

      <!-- Summary -->
      <div v-if="summary" class="summary-box">
        <div class="sum-header">📊 Analysis Summary</div>
        <div class="sum-grid">
          <div class="sum-item">
            <span class="sum-k">Total Words</span>
            <span class="sum-v">{{ summary.total }}</span>
          </div>
          <div class="sum-item">
            <span class="sum-k">✅ Agree</span>
            <span class="sum-v green">{{ summary.agree }}</span>
          </div>
          <div class="sum-item">
            <span class="sum-k">❌ Conflict</span>
            <span class="sum-v red">{{ summary.conflict }}</span>
          </div>
          <div class="sum-item">
            <span class="sum-k">Confidence</span>
            <span class="sum-v" :class="summary.confClass">{{ summary.confidence }}</span>
          </div>
        </div>
      </div>

      <!-- Research Insights -->
      <details class="research-box" open>
        <summary>🔬 Research Insights</summary>
        <div class="research-content">
          <div class="insight-item">🔵 <strong>CAMeL</strong> uses Statistical MLE trained on Penn Arabic Treebank</div>
          <div class="insight-item">🟢 <strong>Stanza</strong> uses Neural BiLSTM with Universal Dependencies</div>
          <div class="insight-item">🟡 <strong>Qalsadi</strong> uses Rule-based dictionary lemmatization (lemma only — no POS)</div>
          <div class="insight-item">🟣 <strong>Farasa</strong> uses SVM-rank for segmentation (98.94% F1)</div>
        </div>
      </details>
    </section>
  </div>
</template>

<script setup>
// Version: 8.3
import { computed, ref } from 'vue'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const evalData = ref(null)
const showDetails = ref(false)

const POS_UNIFIED = {
  NOUN: 'NOUN',
  VERB: 'VERB',
  ADJECTIVE: 'ADJ',
  ADJ: 'ADJ',
  ADPOSITION: 'ADP',
  ADP: 'ADP',
  PRONOUN: 'PRON',
  PRON: 'PRON',
  ADVERB: 'ADV',
  ADV: 'ADV',
  CONJUNCTION: 'CCONJ',
  CCONJ: 'CCONJ',
  PARTICLE: 'PART',
  PART: 'PART',
  PUNCTUATION: 'PUNCT',
  PUNCT: 'PUNCT',
  PRON_REL: 'PRON',
  PROPN: 'NOUN',
  NOUN_PROP: 'NOUN',
  AUX: 'VERB',
  DET: 'PRON',
  SCONJ: 'SCONJ',
  NUM: 'NUM',
  X: 'X',
}

function normalizePos(pos) {
  if (!pos) return null
  return POS_UNIFIED[String(pos).toUpperCase()] || String(pos).toUpperCase()
}

function posAgrees(camelPos, stanzaPos) {
  const c = normalizePos(camelPos)
  const s = normalizePos(stanzaPos)
  return !!(c && s && c === s)
}

async function compare() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  evalData.value = null

  try {
    const [combinedRes, evalRes] = await Promise.all([
      axios.get(`${API}/analyze-combined`, {
        params: { text: inputText.value },
        timeout: 180000,
      }),
      axios.get(`${API}/evaluate`, {
        params: { text: inputText.value },
        timeout: 180000,
      }),
    ])

    results.value = combinedRes.data
    evalData.value = evalRes.data?.evaluation ?? null
  } catch (e) {
    error.value = 'Failed to connect to the backend. Start the FastAPI server and try again.'
    console.error(e)
  } finally {
    loading.value = false
  }
}

const rows = computed(() => {
  if (!results.value) return []

  const camel = results.value?.camel?.tokens || []
  const stanza = results.value?.stanza?.tokens || []
  const farasa = results.value?.farasa?.tokens || []
  const qalsadi = results.value?.qalsadi?.tokens || []

  const max = Math.max(camel.length, stanza.length, farasa.length, qalsadi.length)

  return Array.from({ length: max }, (_, index) => {
    const camelToken = camel[index]
    const camelBest = camelToken?.analyses?.[0] || {}
    const stanzaToken = stanza[index] || {}
    const farasaToken = farasa[index] || {}
    const qalsadiToken = qalsadi[index] || {}

    const word = camelToken?.surface || stanzaToken.surface || farasaToken.surface || qalsadiToken.surface || `#${index + 1}`

    const agrees = posAgrees(camelBest.pos, stanzaToken.upos)
    const reason = !agrees && camelBest.pos && stanzaToken.upos
      ? `CAMeL: ${normalizePos(camelBest.pos)} | Stanza: ${normalizePos(stanzaToken.upos)}`
      : null

    return {
      index,
      word,
      agrees,
      reason,
      camel: { pos: camelBest.pos, lemma: camelBest.lemma, root: camelBest.root },
      stanza: { pos: stanzaToken.upos, lemma: stanzaToken.lemma },
      farasa: { segments: farasaToken.segmentation?.join(' + ') || '' },
      qalsadi: { lemma: qalsadiToken.lemma || null },
    }
  })
})

// Metrics MUST come from /evaluate
const metrics = computed(() => {
  if (!evalData.value) {
    return { posAgreement: 0, lemmaMatch: 0, f1: 0, farasaCoverage: 0 }
  }

  const posAgreement = Math.round(parseFloat(evalData.value.pos_agreement_pct) || 0)
  const lemmaMatch = Math.round(parseFloat(evalData.value.lemma_match_pct) || 0)
  const f1 = Math.round((evalData.value.pos_f1 || 0) * 100)
  const farasaCoverage = Math.round(parseFloat(evalData.value.segmentation_coverage || 0) * 100)

  return { posAgreement, lemmaMatch, f1, farasaCoverage }
})

const summary = computed(() => {
  const r = rows.value
  if (!r.length) return null

  const agree = r.filter(x => x.agrees).length
  const conflict = r.filter(x => !x.agrees && x.camel.pos && x.stanza.pos).length

  const pct = metrics.value.posAgreement
  const confidence = pct >= 75 ? 'High 🟢' : pct >= 50 ? 'Medium 🟡' : 'Low 🔴'
  const confClass = pct >= 75 ? 'green' : pct >= 50 ? 'amber' : 'red'

  return { total: r.length, agree, conflict, confidence, confClass }
})

function rowClass(row) {
  if (!row.camel.pos || !row.stanza.pos) return ''
  return row.agrees ? 'row-ok' : 'row-bad'
}

function metricColor(pct) {
  if (pct >= 80) return 'metric-good'
  if (pct >= 50) return 'metric-mid'
  return 'metric-poor'
}

function posClass(pos) {
  const v = normalizePos(pos)
  if (v === 'VERB') return 'pill pill-blue'
  if (v === 'NOUN') return 'pill pill-green'
  if (v === 'ADJ') return 'pill pill-violet'
  if (v === 'ADP') return 'pill pill-amber'
  if (v === 'PRON') return 'pill pill-purple'
  return 'pill pill-gray'
}

function value(item) {
  return item || '—'
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

.metrics-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:18px; }
.metric { display:flex; flex-direction:column; align-items:center; gap:6px; text-align:center; padding:20px 16px; border-radius:12px; border:1px solid var(--border,#e2e8f0); transition:transform .18s; }
.metric:hover { transform:translateY(-2px); }
.metric-icon { font-size:22px; }
.metric strong { font-size:28px; font-weight:900; color:var(--navy,#1e3a5f); line-height:1; }
.metric span { font-size:13px; font-weight:600; color:var(--muted,#64748b); }
.metric-good { border-top:3px solid #10b981; box-shadow:0 0 0 2px rgba(16,185,129,.1); }
.metric-mid { border-top:3px solid #f59e0b; box-shadow:0 0 0 2px rgba(245,158,11,.1); }
.metric-poor { border-top:3px solid #ef4444; box-shadow:0 0 0 2px rgba(239,68,68,.1); }

.result-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:16px; }
.table-scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:14px; }

th { background:#1e3a5f; color:#fff; padding:10px 14px; text-align:left; font-size:12px; font-weight:700; }
.th-camel { background:#2E5FA3 !important; }
.th-stanza { background:#059669 !important; }
.th-qalsadi { background:#d97706 !important; }
.th-farasa { background:#7C3AED !important; }

tbody tr:nth-child(even):not(.row-ok):not(.row-bad) { background:#f8fafc; }
tbody tr:hover { background:#eff6ff; }

td { padding:10px 14px; border-bottom:1px solid #f1f5f9; vertical-align:middle; }
.word-cell { font-size:17px; font-weight:800; color:#1e3a5f; min-width:80px; }
.segments { color:#7C3AED; font-weight:700; }
.qalsadi-cell { color:#d97706; font-weight:600; font-size:15px; }
.row-ok { background:#f0fdf4 !important; border-left:3px solid #10b981; }
.row-bad { background:#fef2f2 !important; border-left:3px solid #ef4444; }
.detail-row td { background:#f9fafb; color:var(--muted,#64748b); font-size:13px; line-height:1.7; }

.agree-cell { display:flex; flex-direction:column; gap:4px; }
.reason-text { font-size:11px; color:#9f1239; font-style:italic; }

.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.pill-blue { background:#dbeafe; color:#1d4ed8; }
.pill-green { background:#dcfce7; color:#166534; }
.pill-violet { background:#ede9fe; color:#7c3aed; }
.pill-amber { background:#fef3c7; color:#92400e; }
.pill-purple { background:#f5e8fd; color:#8e44ad; }
.pill-gray { background:#f1f5f9; color:#64748b; }

.summary-box { margin-top:20px; padding:20px 24px; border-left:4px solid #2563eb; background:#eff6ff; border-radius:0 12px 12px 0; }
.sum-header { font-size:16px; font-weight:800; color:#1e3a5f; margin-bottom:14px; }
.sum-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.sum-item { display:flex; flex-direction:column; gap:4px; }
.sum-k { font-size:12px; color:#64748b; font-weight:600; }
.sum-v { font-size:16px; font-weight:800; color:#1e3a5f; }
.sum-v.green { color:#059669; }
.sum-v.amber { color:#d97706; }
.sum-v.red { color:#dc2626; }

.research-box { margin-top:20px; }
.research-box summary { font-size:15px; font-weight:800; color:#1e3a5f; cursor:pointer; padding:12px 0; list-style:none; }
.research-content { display:flex; flex-direction:column; gap:8px; margin-top:12px; }
.insight-item { padding:10px 14px; background:#f8fafc; border-radius:8px; font-size:13px; line-height:1.6; }

.loading-state { padding:24px; text-align:center; color:#64748b; font-weight:600; font-size:15px; }
.error-state { padding:14px 16px; background:#fef2f2; border:1px solid #fecaca; border-radius:8px; color:#b91c1c; font-weight:600; margin-top:16px; }

@media (max-width:860px) {
  .metrics-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .input-head { flex-direction:column; }
  .sum-grid { grid-template-columns:repeat(2,1fr); }
}
</style>

