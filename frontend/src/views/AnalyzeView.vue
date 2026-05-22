<template>
  <div class="page-wrap analyze-page">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Analysis workspace</span>
      <h1 class="hero-title">Analyze Arabic Text</h1>
      <p class="hero-copy">
        Submit Arabic text once and inspect the outputs from every NLP tool in a compact,
        research-ready interface.
      </p>
    </section>

    <section class="panel panel-pad input-panel">
      <div class="input-head">
        <div>
          <h2 class="section-title">Input</h2>
          <p class="section-subtitle">Arabic text is sent to the FastAPI backend.</p>
        </div>
        <span class="pill pill-gray">{{ tokenEstimate }} tokens estimated</span>
      </div>

      <textarea
        v-model="inputText"
        class="textarea arabic"
        dir="rtl"
        placeholder="اكتب النص العربي هنا..."
      ></textarea>

      <div class="actions-row form-actions">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="analyze">
          {{ loading ? 'Analyzing...' : 'Run Analysis' }}
        </button>
        <button class="btn btn-secondary" @click="loadSample">Sample</button>
        <button class="btn btn-secondary" @click="clear">Clear</button>
        <button class="btn btn-secondary" :disabled="!results" @click="download('json')">JSON</button>
        <button class="btn btn-secondary" :disabled="!results" @click="download('csv')">CSV</button>
      </div>
    </section>

    <div v-if="loading" class="loading-state">Running CAMeL, Farasa, Stanza, and Qalsadi...</div>
    <div v-if="error" class="error-state">{{ error }}</div>

    <section v-if="results && !loading" class="results-grid">
      <article class="panel panel-pad overview-panel">
        <h2 class="section-title">Run Summary</h2>
        <div class="summary-grid">
          <div v-for="item in summary" :key="item.label" class="summary-item">
            <strong>{{ item.value }}</strong>
            <span>{{ item.label }}</span>
          </div>
        </div>
      </article>

      <article class="panel panel-pad fusion-panel">
        <div class="result-head">
          <div>
            <h2 class="section-title">Fusion Output</h2>
            <p class="section-subtitle">Fusion remains based on CAMeL, Farasa, and Stanza.</p>
          </div>
          <button class="btn btn-secondary" @click="loadFusion">Refresh Fusion</button>
        </div>

        <div v-if="fusionRows.length" class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>Lemma</th>
                <th>Root</th>
                <th>POS</th>
                <th>Segmentation</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in fusionRows" :key="row.word">
                <td class="arabic word-cell" dir="rtl">{{ row.word }}</td>
                <td>{{ value(row.final?.lemma) }}</td>
                <td>{{ value(row.final?.root) }}</td>
                <td><span :class="posClass(row.final?.pos)">{{ value(row.final?.pos) }}</span></td>
                <td>{{ row.final?.segmentation?.join(' + ') || '-' }}</td>
                <td><span class="pill pill-gray">{{ value(row.final?.confidence_level) }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty-state">Fusion output is not available for this run.</div>
      </article>

      <article v-for="tool in toolSections" :key="tool.key" class="panel panel-pad tool-section">
        <div class="result-head">
          <div>
            <h2 class="section-title">{{ tool.name }}</h2>
            <p class="section-subtitle">{{ tool.subtitle }}</p>
          </div>
          <span :class="statusPill(results[tool.key]?.status)">{{ results[tool.key]?.status || 'missing' }}</span>
        </div>

        <div v-if="tool.key === 'farasa'" class="segmentation-list">
          <div v-for="token in results.farasa?.tokens || []" :key="token.surface" class="segment-card">
            <strong class="arabic" dir="rtl">{{ token.surface }}</strong>
            <span>{{ token.segmentation?.join(' + ') || '-' }}</span>
          </div>
        </div>

        <div v-else class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>Lemma</th>
                <th>{{ tool.key === 'qalsadi' ? 'Stem' : 'Root' }}</th>
                <th>POS</th>
                <th>{{ tool.key === 'qalsadi' ? 'Arabic POS' : 'Gender' }}</th>
                <th>{{ tool.key === 'qalsadi' ? 'Unvocalized' : 'Number' }}</th>
                <th>Extra</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in normalizedRows(tool.key)" :key="`${tool.key}-${row.surface}-${row.index}`">
                <td class="arabic word-cell" dir="rtl">{{ row.surface }}</td>
                <td>{{ value(row.lemma) }}</td>
                <td>{{ value(row.root) }}</td>
                <td><span :class="posClass(row.pos)">{{ value(row.pos) }}</span></td>
                <td>{{ value(row.gender) }}</td>
                <td>{{ value(row.number) }}</td>
                <td>{{ row.extra }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { analyzeAll, fusionText } from '../api/nlpApi'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const fusionRows = ref([])

const toolSections = [
  { key: 'camel', name: 'CAMeL Tools', subtitle: 'Top morphological analysis per token.' },
  { key: 'stanza', name: 'Stanza', subtitle: 'Lemma, universal POS, and syntactic dependency.' },
  { key: 'qalsadi', name: 'Qalsadi', subtitle: 'Rule-based lemma, stem, and Arabic POS tags.' },
  { key: 'farasa', name: 'Farasa', subtitle: 'Segmentation and clitic splitting.' },
]

const tokenEstimate = computed(() => inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0)

const summary = computed(() => {
  const data = results.value || {}
  return [
    { label: 'CAMeL tokens', value: data.camel?.tokens?.length || 0 },
    { label: 'Farasa tokens', value: data.farasa?.tokens?.length || 0 },
    { label: 'Stanza words', value: data.stanza?.tokens?.length || 0 },
    { label: 'Qalsadi words', value: data.qalsadi?.tokens?.length || 0 },
  ]
})

async function analyze() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  fusionRows.value = []

  try {
    results.value = await analyzeAll(inputText.value)
    await loadFusion()
  } catch {
    error.value = 'Failed to connect to the backend. Make sure uvicorn is running on port 8000.'
  } finally {
    loading.value = false
  }
}

async function loadFusion() {
  if (!inputText.value.trim()) return
  try {
    const data = await fusionText(inputText.value)
    fusionRows.value = data.fusion_result?.fusion || []
  } catch {
    fusionRows.value = []
  }
}

function normalizedRows(key) {
  const data = results.value?.[key]?.tokens || []

  if (key === 'camel') {
    return data.map((token, index) => {
      const best = token.analyses?.[0] || {}
      return {
        index,
        surface: token.surface,
        lemma: best.lemma,
        root: best.root,
        pos: best.pos,
        gender: best.gender,
        number: best.number,
        extra: best.confidence ? `confidence ${best.confidence}` : '-',
      }
    })
  }

  if (key === 'stanza') {
    return data.map((token, index) => ({
      index,
      surface: token.surface,
      lemma: token.lemma,
      root: null,
      pos: token.upos,
      gender: token.gender,
      number: token.number,
      extra: token.dependency ? `${token.dependency.deprel} -> ${token.dependency.head}` : '-',
    }))
  }

  return data.map((token, index) => ({
    index,
    surface: token.surface,
    lemma: token.lemma,
    root: token.stem,
    pos: token.pos,
    gender: token.pos_ar,
    number: token.unvocalized,
    extra: token.pos_ar ? `Arabic POS: ${token.pos_ar}` : '-',
  }))
}

function statusPill(status) {
  if (status === 'ok') return 'pill pill-green'
  if (status === 'failed') return 'pill pill-amber'
  return 'pill pill-red'
}

function posClass(pos) {
  const value = String(pos || '').toUpperCase()
  if (value.includes('VERB')) return 'pill pill-blue'
  if (value.includes('NOUN')) return 'pill pill-green'
  if (value.includes('ADJ')) return 'pill pill-violet'
  if (value.includes('ADP')) return 'pill pill-amber'
  return 'pill pill-gray'
}

function value(item) {
  return item || '-'
}

function clear() {
  inputText.value = ''
  results.value = null
  fusionRows.value = []
  error.value = ''
}

function loadSample() {
  inputText.value = 'ذهب محمد إلى جامعة بيرزيت لدراسة اللغة العربية'
}

function download(format) {
  const url = `${API}/export?text=${encodeURIComponent(inputText.value)}&format=${format}`
  window.open(url, '_blank', 'noopener,noreferrer')
}

onMounted(() => {
  if (route.query.text) {
    inputText.value = String(route.query.text)
    analyze()
  }
})
</script>

<style scoped>
.compact-hero {
  padding: 34px 38px;
}

.compact-hero .hero-title {
  font-size: 38px;
}

.input-panel {
  margin-top: 18px;
}

.input-head,
.result-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  margin-bottom: 16px;
}

.form-actions {
  margin-top: 16px;
}

.results-grid {
  display: grid;
  gap: 18px;
  margin-top: 18px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.summary-item {
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.summary-item strong,
.summary-item span {
  display: block;
}

.summary-item strong {
  font-size: 28px;
  color: var(--navy);
}

.summary-item span {
  margin-top: 4px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.word-cell {
  font-size: 18px;
  font-weight: 800;
}

.segmentation-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.segment-card {
  display: grid;
  gap: 8px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.segment-card span {
  color: var(--violet);
  font-weight: 800;
}

@media (max-width: 760px) {
  .input-head,
  .result-head {
    flex-direction: column;
  }

  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
