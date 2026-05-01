<template>
  <div class="container">
    <h1 class="page-title">Analyze Arabic Text</h1>

    <!-- Input -->
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
          @click="analyze"
          :disabled="loading || !inputText.trim()"
        >
          {{ loading ? 'جاري التحليل...' : 'Analyze' }}
        </button>
        <button class="btn btn-clear" @click="clear">Clear</button>
      </div>
    </div>

    <div v-if="loading" class="loading"><div class="spinner"></div> Analyzing...</div>
    <div v-if="error" class="error-msg">{{ error }}</div>

    <!-- Results -->
    <div v-if="results && !loading">
      <!-- Export Buttons -->
      <div class="export-bar">
        <button class="btn btn-export" @click="exportJSON">Download JSON</button>
        <button class="btn btn-export" @click="exportCSV">Download CSV</button>
      </div>

      <!-- CAMeL Card -->
      <div class="card result-card camel-card">
        <div class="card-header camel-header">
          <span class="tool-badge">🔵 CAMeL Tools</span>
          <span class="tool-subtitle">Morphological Analysis</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>Lemma</th>
                <th>Root</th>
                <th>POS</th>
                <th>Gender</th>
                <th>Number</th>
                <th>Tense</th>
                <th>Gloss</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="token in results.camel?.tokens" :key="token.surface">
                <tr v-for="(a, i) in token.analyses" :key="i">
                  <td class="arabic cell-word">{{ i === 0 ? token.surface : '' }}</td>
                  <!-- Lemma: RTL for Arabic, LTR for proper rendering -->
                  <td class="arabic" dir="ltr" style="text-align:center" :class="getDisagreeClass('lemma', token.surface, a.lemma)">{{ cleanValue(a.lemma) || '—' }}</td>
                  <!-- FIX BUG 2: Root with LTR direction and cleaning -->
                  <td dir="ltr" style="text-align:center" :class="getDisagreeClass('root', token.surface, a.root)">{{ cleanValue(a.root) || '—' }}</td>
                  <td :class="getDisagreeClass('pos', token.surface, a.pos)">
                    <span :class="posBadge(a.pos)">{{ a.pos || '—' }}</span>
                  </td>
                  <td>{{ a.gender || '—' }}</td>
                  <td>{{ a.number || '—' }}</td>
                  <td>{{ a.tense || '—' }}</td>
                  <td>{{ a.gloss || '—' }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Farasa Card -->
      <div class="card result-card farasa-card">
        <div class="card-header farasa-header">
          <span class="tool-badge">🟣 Farasa</span>
          <span class="tool-subtitle">Segmentation</span>
        </div>
        
        <!-- Show full segmented text at top -->
        <div v-if="results.farasa?.segmented_text" class="full-segmented">
          <span class="seg-label">التقطيع الكامل:</span>
          <span class="arabic seg-full">{{ results.farasa.segmented_text }}</span>
        </div>
        
        <!-- Token segmentation display -->
        <div class="farasa-tokens">
          <div v-for="token in results.farasa?.tokens" :key="token.surface" class="farasa-token">
            <div class="token-surface arabic">{{ token.surface }}</div>
            <div class="morphemes">
              <span 
                v-for="(morpheme, idx) in token.segmentation" 
                :key="idx" 
                class="morpheme-badge"
              >
                {{ morpheme }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Stanza Card -->
      <div class="card result-card stanza-card">
        <div class="card-header stanza-header">
          <span class="tool-badge">🟢 Stanza</span>
          <span class="tool-subtitle">Dependency Parsing</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Word</th>
                <th>Lemma</th>
                <th>Root</th>
                <th>POS</th>
                <th>Gender</th>
                <th>Number</th>
                <th>Tense</th>
                <th>Gloss</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="token in results.stanza?.tokens" :key="token.surface">
                <td class="arabic cell-word">{{ token.surface }}</td>
                <!-- Lemma with LTR for proper rendering -->
                <td class="arabic" dir="ltr" style="text-align:center" :class="getDisagreeClass('lemma', token.surface, token.lemma)">{{ cleanValue(token.lemma) || '—' }}</td>
                <!-- Stanza doesn't have root, show dash -->
                <td>—</td>
                <td :class="getDisagreeClass('pos', token.surface, token.upos)">
                  <span :class="posBadge(token.upos)">{{ token.upos || '—' }}</span>
                </td>
                <td>{{ token.gender || '—' }}</td>
                <td>{{ token.number || '—' }}</td>
                <td>{{ token.tense || '—' }}</td>
                <td>{{ token.gloss || '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)

// disagreement detection
const allAnalyses = ref({})

onMounted(() => {
  if (route.query.text) {
    inputText.value = route.query.text
    analyze()
  }
})

watch(() => route.query.text, (newText) => {
  if (newText) {
    inputText.value = newText
    analyze()
  }
})

async function analyze() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  try {
    const res = await axios.get('http://127.0.0.1:8000/analyze-combined', {
      params: { text: inputText.value }
    })
    results.value = res.data
    detectDisagreements(res.data)
  } catch (e) {
    error.value = 'Failed to connect to backend. Make sure the server is running.'
  } finally {
    loading.value = false
  }
}

function detectDisagreements(data) {
  const analyses = {}
  
  // Collect all lemmas, roots, POS per surface form
  if (data.camel?.tokens) {
    data.camel.tokens.forEach(t => {
      if (!analyses[t.surface]) analyses[t.surface] = {}
      if (t.analyses?.[0]) {
        analyses[t.surface].camel = {
          lemma: t.analyses[0].lemma,
          root: t.analyses[0].root,
          pos: t.analyses[0].pos
        }
      }
    })
  }
  
  if (data.farasa?.tokens) {
    data.farasa.tokens.forEach(t => {
      if (!analyses[t.surface]) analyses[t.surface] = {}
      if (t.analyses?.[0]) {
        analyses[t.surface].farasa = {
          lemma: t.analyses[0].lemma,
          root: t.analyses[0].root,
          pos: t.analyses[0].pos
        }
      }
    })
  }
  
  if (data.stanza?.tokens) {
    data.stanza.tokens.forEach(t => {
      if (!analyses[t.surface]) analyses[t.surface] = {}
      analyses[t.surface].stanza = {
        lemma: t.lemma,
        root: null,
        pos: t.upos
      }
    })
  }
  
  allAnalyses.value = analyses
}

function getDisagreeClass(field, surface, value) {
  if (!allAnalyses.value[surface]) return ''
  
  const surfaceAnalyses = allAnalyses.value[surface]
  const values = []
  
  if (surfaceAnalyses.camel?.[field]) values.push(surfaceAnalyses.camel[field])
  if (surfaceAnalyses.farasa?.[field]) values.push(surfaceAnalyses.farasa[field])
  if (surfaceAnalyses.stanza?.[field]) values.push(surfaceAnalyses.stanza[field])
  
  // Unique non-empty values
  const uniqueValues = [...new Set(values.filter(v => v && v !== '—'))]
  
  if (uniqueValues.length > 1) {
    return 'cell-disagree'
  }
  return ''
}

// FIX BUG 2: Clean root/lemma value - remove annotation artifacts
function cleanValue(value) {
  if (!value) return null
  // Convert to string
  let cleaned = String(value)
  // Remove trailing/leading non-Arabic non-dot characters (like 'A', annotations)
  // Keep only Arabic letters, dots, and common Arabic root pattern characters
  cleaned = cleaned.replace(/^[^\u0600-\u06FF.]+/, '').replace(/[^\u0600-\u06FF.]+$/, '')
  // If empty after cleaning, return null
  return cleaned || null
}

function clear() {
  inputText.value = ''
  results.value = null
  error.value = ''
  allAnalyses.value = {}
}

async function exportJSON() {
  if (!inputText.value.trim()) return
  try {
    const res = await axios.get('http://127.0.0.1:8000/export', {
      params: { text: inputText.value, format: 'json' }
    })
    downloadFile(JSON.stringify(res.data, null, 2), 'analysis.json', 'application/json')
  } catch (e) {
    error.value = 'Failed to export JSON.'
  }
}

async function exportCSV() {
  if (!inputText.value.trim()) return
  try {
    const res = await axios.get('http://127.0.0.1:8000/export', {
      params: { text: inputText.value, format: 'csv' },
      responseType: 'blob'
    })
    const blob = new Blob([res.data], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'analysis.csv'
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    error.value = 'Failed to export CSV.'
  }
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function posBadge(pos) {
  const map = {
    VERB: 'badge badge-blue',
    NOUN: 'badge badge-green',
    ADJ: 'badge badge-purple',
    ADJECTIVE: 'badge badge-purple',
    ADPOSITION: 'badge badge-gray',
    PRONOUN: 'badge badge-gray',
    CONJUNCTION: 'badge badge-gray',
    PARTICLE: 'badge badge-gray',
    PROPN: 'badge badge-purple'
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

.export-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.btn-export {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  background: #1F3864;
  color: white;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-export:hover {
  background: #2E5FA3;
}

.result-card {
  margin-bottom: 24px;
  overflow: hidden;
}

.card-header {
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.camel-header {
  background: #2E5FA3;
}

.farasa-header {
  background: #6C3483;
}

.stanza-header {
  background: #1E8449;
}

.tool-badge {
  font-size: 1rem;
  font-weight: 700;
  color: white;
}

.tool-subtitle {
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.85);
}

.table-container {
  overflow-x: auto;
}

.cell-word {
  font-weight: 700;
  min-width: 100px;
}

.cell-disagree {
  background: #FADBD8 !important;
  position: relative;
}

.cell-disagree::after {
  content: '⚠';
  position: absolute;
  right: 4px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.7rem;
  color: #922B21;
}

/* Farasa segmentation display */
.full-segmented {
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.seg-label {
  font-weight: 600;
  color: #5D6D7E;
  margin-right: 8px;
}

.seg-full {
  color: #6C3483;
  font-weight: 600;
}

.farasa-tokens {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 16px;
}

.farasa-token {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #f8fafc;
  border-radius: 8px;
  min-width: 80px;
}

.token-surface {
  font-size: 1.1rem;
  font-weight: 700;
  color: #1F3864;
}

.morphemes {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  justify-content: center;
}

.morpheme-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  background: #6C3483;
  color: white;
  font-size: 0.85rem;
  font-weight: 600;
}
</style>
