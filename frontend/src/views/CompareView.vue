<template>
  <div class="container">
    <h1 class="page-title">Compare Tools</h1>

    <div class="card input-card">
      <label class="input-label">Enter Arabic Text:</label>
      <textarea
        v-model="inputText"
        class="arabic text-input"
        dir="rtl"
        placeholder="اكتب النص العربي هنا..."
        rows="4"
      ></textarea>
      <div class="input-actions">
        <button
          class="btn btn-primary"
          @click="compare"
          :disabled="loading || !inputText.trim()"
        >
          {{ loading ? 'جاري المقارنة...' : 'Compare' }}
        </button>
        <button class="btn btn-clear" @click="clear">Clear</button>
      </div>
    </div>

    <div v-if="loading" class="loading"><div class="spinner"></div> Comparing...</div>
    <div v-if="error" class="error-msg">{{ error }}</div>

    <!-- Comparison Table -->
    <div v-if="results && !loading" class="card">
      <h2 class="tool-title">Token-by-Token Comparison</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Word</th>
              <th class="th-camel">CAMeL Lemma</th>
              <th class="th-camel">CAMeL POS</th>
              <th class="th-stanza">Stanza Lemma</th>
              <th class="th-stanza">Stanza UPOS</th>
              <th class="th-farasa">Farasa Segments</th>
              <th>Agreement?</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in compareRows" :key="row.surface">
              <td class="arabic surface-cell">{{ row.surface }}</td>
              <!-- CAMeL -->
              <td class="arabic">{{ row.camel?.lemma || '—' }}</td>
              <td>
                <span :class="posBadge(row.camel?.pos)">{{ row.camel?.pos || '—' }}</span>
              </td>
              <!-- Stanza -->
              <td class="arabic">{{ row.stanza?.lemma || '—' }}</td>
              <td>
                <span :class="posBadge(row.stanza?.upos)">{{ row.stanza?.upos || '—' }}</span>
              </td>
              <!-- Farasa -->
              <td>
                <span class="farasa-seg">{{ row.farasa_segments || '—' }}</span>
              </td>
              <!-- Agreement -->
              <td>
                <span :class="row.agree ? 'agree-badge agree-yes' : 'agree-badge agree-no'">
                  {{ row.agree ? '✅ Agree' : '❌ Disagree' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Evaluation Metrics - FIX BUG 1: No double % -->
    <div v-if="metrics && !loading" class="metrics-grid">
      <div class="card metric-card metric-pos">
        <div class="metric-icon">🎯</div>
        <!-- FIX: Use directly, no extra % -->
        <div class="metric-value">{{ metrics.pos_agreement_pct || '—' }}</div>
        <div class="metric-label">POS Agreement</div>
      </div>
      <div class="card metric-card metric-lemma">
        <div class="metric-icon">📖</div>
        <!-- FIX: Use directly, no extra % -->
        <div class="metric-value">{{ metrics.lemma_match_pct || '—' }}</div>
        <div class="metric-label">Lemma Match</div>
      </div>
      <div class="card metric-card metric-f1">
        <div class="metric-icon">📊</div>
        <!-- FIX: pos_f1 is decimal, multiply by 100 -->
        <div class="metric-value">{{ formatF1(metrics.pos_f1) }}</div>
        <div class="metric-label">F1 Score</div>
      </div>
      <div class="card metric-card metric-coverage">
        <div class="metric-icon">✅</div>
        <!-- FIX: segmentation_coverage is decimal, multiply by 100 -->
        <div class="metric-value">{{ formatCoverage(metrics.segmentation_coverage) }}</div>
        <div class="metric-label">Segmentation Coverage</div>
      </div>
    </div>
    
    <!-- Loading state for metrics -->
    <div v-else-if="loading" class="metrics-grid">
      <div class="card metric-card">
        <div class="metric-loading">Loading...</div>
      </div>
      <div class="card metric-card">
        <div class="metric-loading">Loading...</div>
      </div>
      <div class="card metric-card">
        <div class="metric-loading">Loading...</div>
      </div>
      <div class="card metric-card">
        <div class="metric-loading">Loading...</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const metrics = ref(null)

onMounted(() => {
  if (route.query.text) {
    inputText.value = route.query.text
    compare()
  }
})

watch(() => route.query.text, (newText) => {
  if (newText) {
    inputText.value = newText
    compare()
  }
})

async function compare() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  metrics.value = null

  try {
    // Get analysis results
    const [analysisRes, metricsRes] = await Promise.all([
      axios.get('http://127.0.0.1:8000/analyze-combined', {
        params: { text: inputText.value }
      }),
      axios.get('http://127.0.0.1:8000/evaluate', {
        params: { text: inputText.value }
      })
    ])

    results.value = analysisRes.data
    // Map evaluation response fields
    const evalData = metricsRes.data.evaluation || metricsRes.data
    metrics.value = {
      pos_agreement_pct: evalData.pos_agreement_pct,
      lemma_match_pct: evalData.lemma_match_pct,
      pos_f1: evalData.pos_f1,
      segmentation_coverage: evalData.segmentation_coverage
    }
  } catch (e) {
    error.value = 'Failed to connect to backend.'
  } finally {
    loading.value = false
  }
}

function clear() {
  inputText.value = ''
  results.value = null
  metrics.value = null
  error.value = ''
}

const compareRows = computed(() => {
  if (!results.value) return []

  const camelTokens = results.value.camel?.tokens || []
  const farasaTokens = results.value.farasa?.tokens || []
  const stanzaTokens = results.value.stanza?.tokens || []

  return camelTokens.map((ct, i) => {
    const camelPOS = ct.analyses?.[0]?.pos || ''
    const stanzaPOS = stanzaTokens[i]?.upos || ''
    const agree = normalizePOS(camelPOS) === normalizePOS(stanzaPOS)
    
    // Get segmentation array and join with "+"
    const farasaSeg = farasaTokens[i]?.segmentation
    const farasaSegmentsStr = (farasaSeg && farasaSeg.length > 0) 
      ? farasaSeg.join('+') 
      : ''

    return {
      surface: ct.surface,
      camel: ct.analyses?.[0] || null,
      farasa_segments: farasaSegmentsStr,
      stanza: stanzaTokens[i] || null,
      agree
    }
  })
})

function normalizePOS(pos) {
  if (!pos) return ''
  const map = {
    'VERB': 'VERB',
    'NOUN': 'NOUN',
    'ADJ': 'ADJECTIVE',
    'ADJECTIVE': 'ADJECTIVE'
  }
  return map[pos] || pos
}

// FIX BUG 1: Format F1 score - pos_f1 is decimal (0.667), multiply by 100
function formatF1(f1) {
  if (f1 === undefined || f1 === null) return '—'
  return (f1 * 100).toFixed(1) + '%'
}

// FIX BUG 1: Format coverage - segmentation_coverage is decimal (1.0), multiply by 100
function formatCoverage(coverage) {
  if (coverage === undefined || coverage === null) return '—'
  return (coverage * 100).toFixed(1) + '%'
}

function posBadge(pos) {
  const map = {
    VERB: 'badge badge-blue',
    NOUN: 'badge badge-green',
    ADJ: 'badge badge-purple',
    ADJECTIVE: 'badge badge-purple'
  }
  return map[pos] || 'badge badge-gray'
}
</script>

<style scoped>
.page-title {
  font-size: 1.5rem;
  color: #1F3864;
  margin-bottom: 20px;
}

.input-card {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2f6 100%);
}

.input-label {
  display: block;
  font-weight: 600;
  color: #1F3864;
  margin-bottom: 8px;
}

.text-input {
  width: 100%;
  padding: 14px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1.2rem;
  resize: vertical;
  outline: none;
  transition: border 0.2s;
}

.text-input:focus {
  border-color: #2E5FA3;
}

.input-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.btn-clear {
  padding: 10px 20px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  font-weight: 600;
  color: #5D6D7E;
  transition: all 0.2s;
}

.btn-clear:hover {
  border-color: #5D6D7E;
}

.tool-title {
  font-size: 1.1rem;
  color: #1F3864;
  margin-bottom: 16px;
}

.table-wrap {
  overflow-x: auto;
}

.th-camel {
  background: #2E5FA3 !important;
}

.th-stanza {
  background: #1E8449 !important;
}

.th-farasa {
  background: #6C3483 !important;
}

.surface-cell {
  font-weight: 700;
  color: #1F3864;
  min-width: 80px;
}

.farasa-seg {
  font-family: 'Traditional Arabic', 'Arial Unicode MS', sans-serif;
  color: #6C3483;
  font-weight: 600;
}

.agree-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
}

.agree-yes {
  background: #D5F5E3;
  color: #1E8449;
}

.agree-no {
  background: #FADBD8;
  color: #922B21;
}

/* Metrics Cards */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-top: 24px;
}

.metric-card {
  text-align: center;
  padding: 24px 16px;
}

.metric-pos {
  border-top: 4px solid #2E5FA3;
}

.metric-lemma {
  border-top: 4px solid #6C3483;
}

.metric-f1 {
  border-top: 4px solid #1E8449;
}

.metric-coverage {
  border-top: 4px solid #F39C12;
}

.metric-icon {
  font-size: 1.8rem;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 1.8rem;
  font-weight: 700;
  color: #1F3864;
  margin-bottom: 4px;
}

.metric-label {
  font-size: 0.85rem;
  color: #5D6D7E;
}

.metric-loading {
  color: #5D6D7E;
  font-size: 1rem;
}

@media (max-width: 900px) {
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 500px) {
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
</style>
