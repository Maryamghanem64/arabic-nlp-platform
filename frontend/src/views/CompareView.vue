<template>
  <div class="page-wrap compare-page">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Comparison view</span>
      <h1 class="hero-title">Compare Tool Decisions</h1>
      <p class="hero-copy">
        Review token-level differences between CAMeL, Stanza, Farasa, and Qalsadi without
        requiring a separate evaluation endpoint.
      </p>
    </section>

    <section class="panel panel-pad input-panel">
      <div class="input-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
          <p class="section-subtitle">The comparison uses the combined analysis endpoint.</p>
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

    <div v-if="loading" class="loading-state">Comparing token outputs...</div>
    <div v-if="error" class="error-state">{{ error }}</div>

    <section v-if="results && !loading" class="comparison-results">
      <div class="metrics-grid">
        <article class="panel panel-pad metric">
          <strong>{{ metrics.posAgreement }}%</strong>
          <span>POS agreement</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.lemmaAgreement }}%</strong>
          <span>Lemma agreement</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.tokens }}</strong>
          <span>Compared tokens</span>
        </article>
        <article class="panel panel-pad metric">
          <strong>{{ metrics.qalsadiTokens }}</strong>
          <span>Qalsadi tokens</span>
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
                <th>Qalsadi POS</th>
                <th>Farasa Segments</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="row in rows" :key="row.index">
                <tr>
                  <td class="arabic word-cell" dir="rtl">{{ row.word }}</td>
                  <td><span :class="posClass(row.camel.pos)">{{ value(row.camel.pos) }}</span></td>
                  <td><span :class="posClass(row.stanza.pos)">{{ value(row.stanza.pos) }}</span></td>
                  <td><span :class="posClass(row.qalsadi.pos)">{{ value(row.qalsadi.pos) }}</span></td>
                  <td class="segments">{{ row.farasa.segments || '-' }}</td>
                  <td>
                    <span :class="row.agrees ? 'pill pill-green' : 'pill pill-red'">
                      {{ row.agrees ? 'Agree' : 'Review' }}
                    </span>
                  </td>
                </tr>
                <tr v-if="showDetails" class="detail-row">
                  <td></td>
                  <td>
                    <strong>Lemma:</strong> {{ value(row.camel.lemma) }}<br />
                    <strong>Root:</strong> {{ value(row.camel.root) }}
                  </td>
                  <td>
                    <strong>Lemma:</strong> {{ value(row.stanza.lemma) }}<br />
                    <strong>Root:</strong> -
                  </td>
                  <td>
                    <strong>Lemma:</strong> {{ value(row.qalsadi.lemma) }}<br />
                    <strong>Stem:</strong> {{ value(row.qalsadi.stem) }}
                  </td>
                  <td colspan="2">
                    <strong>Arabic POS:</strong> {{ value(row.qalsadi.posAr) }}
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const showDetails = ref(false)

async function compare() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null

  try {
    const { data } = await axios.get(`${API}/analyze-combined`, {
      params: { text: inputText.value },
      timeout: 120000,
    })
    results.value = data
  } catch {
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
    const camelToken = camel[index]
    const camelBest = camelToken?.analyses?.[0] || {}
    const stanzaToken = stanza[index] || {}
    const farasaToken = farasa[index] || {}
    const qalsadiToken = qalsadi[index] || {}
    const word = camelToken?.surface || stanzaToken.surface || farasaToken.surface || qalsadiToken.surface || `#${index + 1}`
    const posValues = [camelBest.pos, stanzaToken.upos, qalsadiToken.pos].filter(Boolean).map(normalizePos)
    const uniquePos = [...new Set(posValues)]

    return {
      index,
      word,
      camel: { pos: camelBest.pos, lemma: camelBest.lemma, root: camelBest.root },
      stanza: { pos: stanzaToken.upos, lemma: stanzaToken.lemma },
      farasa: { segments: farasaToken.segmentation?.join(' + ') },
      qalsadi: {
        pos: qalsadiToken.pos,
        lemma: qalsadiToken.lemma,
        stem: qalsadiToken.stem,
        posAr: qalsadiToken.pos_ar,
      },
      agrees: uniquePos.length <= 1 && uniquePos.length > 0,
    }
  })
})

const metrics = computed(() => {
  const currentRows = rows.value
  const comparablePos = currentRows.filter((row) => [row.camel.pos, row.stanza.pos, row.qalsadi.pos].filter(Boolean).length > 1)
  const posAgree = comparablePos.filter((row) => row.agrees).length

  const comparableLemma = currentRows.filter((row) => [row.camel.lemma, row.stanza.lemma, row.qalsadi.lemma].filter(Boolean).length > 1)
  const lemmaAgree = comparableLemma.filter((row) => {
    const values = [row.camel.lemma, row.stanza.lemma, row.qalsadi.lemma].filter(Boolean).map((x) => String(x).trim())
    return new Set(values).size <= 1
  }).length

  return {
    posAgreement: comparablePos.length ? Math.round((posAgree / comparablePos.length) * 100) : 0,
    lemmaAgreement: comparableLemma.length ? Math.round((lemmaAgree / comparableLemma.length) * 100) : 0,
    tokens: currentRows.length,
    qalsadiTokens: currentRows.filter((row) => row.qalsadi.lemma || row.qalsadi.pos).length,
  }
})

function normalizePos(pos) {
  const value = String(pos || '').toUpperCase()
  if (value === 'ADJECTIVE') return 'ADJ'
  if (value === 'ADPOSITION') return 'ADP'
  if (value === 'PRONOUN') return 'PRON'
  return value
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
  error.value = ''
}

function loadSample() {
  inputText.value = 'قرأ الطلاب الكتب في المكتبة'
}
</script>

<style scoped>
.compact-hero {
  padding: 34px 38px;
}

.compact-hero .hero-title {
  font-size: 38px;
}

.input-panel,
.comparison-results {
  margin-top: 18px;
}

.input-head,
.result-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 16px;
}

.toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  color: var(--muted);
  font-size: 14px;
  font-weight: 800;
}

.form-actions {
  margin-top: 16px;
}

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

.word-cell {
  font-size: 18px;
  font-weight: 900;
}

.segments {
  color: var(--violet);
  font-weight: 800;
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
