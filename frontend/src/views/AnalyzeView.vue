<template>
  <div class="page-wrap analyze-page page-stack">
    <section class="hero-band analyze-hero">
      <div class="hero-content">
        <span class="eyebrow">Single tool deep analysis</span>
        <h1 class="hero-title">Inspect every analyzer without crashing when tools are missing.</h1>
        <p class="hero-copy">
          Tool cards show live status from the response. Unavailable analyzers render as
          warning states with clear setup reasons instead of empty output.
        </p>
      </div>
    </section>

    <section class="panel panel-pad">
      <div class="section-head">
        <div>
          <h2 class="section-title">Analysis Input</h2>
          <p class="section-subtitle">{{ tokenEstimate }} token{{ tokenEstimate === 1 ? '' : 's' }} estimated</p>
        </div>
        <div class="actions-row compact-actions">
          <button class="btn btn-subtle" @click="loadSample">Sample</button>
          <button class="btn btn-subtle" @click="clear">Clear</button>
        </div>
      </div>

      <textarea v-model="inputText" class="textarea arabic" dir="rtl" placeholder="اكتب النص العربي هنا..."></textarea>

      <div class="run-row">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="analyze">
          {{ loading ? 'Analyzing...' : 'Run Deep Analysis' }}
        </button>
        <button class="btn btn-secondary" :disabled="!results" @click="copyCurrentJson">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !results }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !results }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

    <div v-if="loading" class="loading-state analysis-loading">
      <div class="loading-stack">
        <span class="skeleton"></span>
        <span class="skeleton wide"></span>
        <span class="skeleton short"></span>
      </div>
    </div>

    <div v-if="error" class="error-state">
      <div>
        <strong>Analysis failed</strong>
        <p>{{ error }}</p>
        <button class="btn btn-secondary" @click="analyze">Retry</button>
      </div>
    </div>

    <template v-if="results && !loading">
      <section class="tool-overview">
        <article
          v-for="tool in toolSections"
          :key="tool.key"
          :class="['panel panel-pad tool-summary', { active: activeTool === tool.key, unavailable: !isToolUsable(tool.key) }]"
          @click="activeTool = tool.key"
        >
          <div class="tool-summary-head">
            <span class="tool-code">{{ tool.code }}</span>
            <span :class="['pill', statusPill(toolStatus(tool.key))]">{{ readableStatus(toolStatus(tool.key)) }}</span>
          </div>
          <h3>{{ tool.name }}</h3>
          <p>{{ tool.subtitle }}</p>
          <small v-if="toolReason(tool.key)" class="tool-reason">{{ toolReason(tool.key) }}</small>
          <strong>{{ tokenCount(tool.key) }}</strong>
        </article>
      </section>

      <section class="panel panel-pad">
        <div class="analysis-tabs" role="tablist" aria-label="Analysis views">
          <button v-for="tab in tabs" :key="tab.key" type="button" :class="{ active: activeTab === tab.key }" @click="activeTab = tab.key">
            {{ tab.label }}
          </button>
        </div>

        <div v-if="activeTab === 'tokens'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">{{ activeToolMeta.name }} Token Breakdown</h2>
              <p class="section-subtitle">{{ activeToolMeta.subtitle }}</p>
            </div>
          </div>

          <div v-if="!isToolUsable(activeTool)" class="warning-state">
            <strong>{{ activeToolMeta.name }} unavailable on this machine</strong>
            <p>{{ toolReason(activeTool) || 'The backend returned a safe unavailable response.' }}</p>
          </div>

          <div v-else-if="activeRows.length" class="token-grid">
            <article v-for="row in activeRows" :key="`${activeTool}-${row.index}`" class="token-detail-card">
              <div class="token-detail-head">
                <strong class="arabic" dir="rtl">{{ row.surface }}</strong>
                <span :class="posClass(row.pos)">{{ value(row.pos) }}</span>
              </div>
              <dl>
                <div><dt>Lemma</dt><dd class="arabic" dir="rtl">{{ value(row.lemma) }}</dd></div>
                <div><dt>{{ activeTool === 'farasa' ? 'Segments' : 'Root / stem' }}</dt><dd>{{ value(row.root) }}</dd></div>
                <div><dt>Extra</dt><dd>{{ row.extra }}</dd></div>
              </dl>
            </article>
          </div>
          <div v-else class="empty-state">No token output is available for this tool.</div>
        </div>

        <div v-if="activeTab === 'fusion'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Fusion Output</h2>
              <p class="section-subtitle">Fusion remains available when enough core tool evidence exists.</p>
            </div>
            <button class="btn btn-secondary" @click="loadFusion">Refresh Fusion</button>
          </div>
          <div v-if="fusionRows.length" class="table-scroll">
            <table>
              <thead><tr><th>Token</th><th>Lemma</th><th>Root</th><th>POS</th><th>Segmentation</th><th>Confidence</th></tr></thead>
              <tbody>
                <tr v-for="row in fusionRows" :key="`${row.word}-${row.final?.pos}`">
                  <td class="arabic fusion-word" dir="rtl">{{ row.word }}</td>
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
        </div>

        <div v-if="activeTab === 'json'" class="tab-panel">
          <div class="section-head">
            <div><h2 class="section-title">Formatted JSON</h2><p class="section-subtitle">Raw response for reproducibility and debugging.</p></div>
          </div>
          <pre class="json-panel"><code>{{ prettyJson }}</code></pre>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { analyzeAll, exportUrl, fusionText } from '../api/nlpApi'

const route = useRoute()
const inputText = ref('')
const loading = ref(false)
const error = ref('')
const results = ref(null)
const fusionRows = ref([])
const activeTool = ref('camel')
const activeTab = ref('tokens')
const copied = ref(false)

const toolSections = [
  { key: 'camel', code: 'CM', name: 'CAMeL Tools', subtitle: 'Morphology, lemma, root, POS, and confidence.' },
  { key: 'farasa', code: 'FA', name: 'Farasa', subtitle: 'Segmentation and clitic splitting evidence.' },
  { key: 'stanza', code: 'ST', name: 'Stanza', subtitle: 'Universal POS, lemmas, and dependencies.' },
  { key: 'qalsadi', code: 'QA', name: 'Qalsadi', subtitle: 'Rule-based lemma evidence.' },
  { key: 'arabert', code: 'AB', name: 'AraBERT', subtitle: 'Optional transformer model integration.' },
  { key: 'alkhalil', code: 'AK', name: 'AlKhalil', subtitle: 'Optional Java morphological analyzer.' },
  { key: 'udpipe', code: 'UD', name: 'UDPipe', subtitle: 'Optional UD parser integration.' },
  { key: 'madamira', code: 'MA', name: 'MADAMIRA', subtitle: 'Optional Java analyzer integration.' },
  { key: 'sinatools', code: 'SI', name: 'SinaTools', subtitle: 'Future microservice analyzer.' },
]

const tabs = [{ key: 'tokens', label: 'Token Breakdown' }, { key: 'fusion', label: 'Fusion' }, { key: 'json', label: 'JSON' }]
const tokenEstimate = computed(() => inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0)
const activeToolMeta = computed(() => toolSections.find((tool) => tool.key === activeTool.value) || toolSections[0])
const jsonExportHref = computed(() => results.value ? exportUrl(inputText.value, 'json') : '#')
const csvExportHref = computed(() => results.value ? exportUrl(inputText.value, 'csv') : '#')
const prettyJson = computed(() => JSON.stringify({ combined: results.value, fusion: fusionRows.value }, null, 2))
const activeRows = computed(() => normalizedRows(activeTool.value))

async function analyze() {
  if (!inputText.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = null
  fusionRows.value = []
  copied.value = false
  try {
    results.value = await analyzeAll(inputText.value)
    await loadFusion()
  } catch (e) {
    error.value = e.message || 'Failed to connect to the backend.'
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
  if (key === 'camel') return data.map((token, index) => {
    const best = token.analyses?.[0] || {}
    return { index, surface: token.surface, lemma: best.lemma, root: best.root, pos: best.pos, extra: best.confidence ? `confidence ${best.confidence}` : '-' }
  })
  if (key === 'farasa') return data.map((token, index) => ({ index, surface: token.surface, lemma: null, root: Array.isArray(token.segmentation) ? token.segmentation.join(' + ') : '-', pos: null, extra: token.segmentation?.length ? `${token.segmentation.length} segment(s)` : '-' }))
  if (key === 'stanza') return data.map((token, index) => ({ index, surface: token.surface, lemma: token.lemma, root: '-', pos: token.upos, extra: token.dependency ? `${token.dependency.deprel} -> ${token.dependency.head_text || token.dependency.head || 'root'}` : '-' }))
  return data.map((token, index) => ({ index, surface: token.surface, lemma: token.lemma, root: token.stem, pos: token.pos, extra: token.reason || 'Safe fallback analyzer' }))
}

function toolStatus(key) {
  return results.value?.[key]?.status || 'missing'
}

function toolReason(key) {
  return results.value?.[key]?.reason || results.value?.[key]?.error || ''
}

function isToolUsable(key) {
  return toolStatus(key) === 'ok'
}

function tokenCount(key) {
  const status = toolStatus(key)
  if (status !== 'ok') return readableStatus(status)
  return `${results.value?.[key]?.tokens?.length || 0} tokens`
}

function readableStatus(status) {
  if (status === 'ok') return 'Online'
  if (status === 'future_work') return 'Planned'
  if (status === 'unavailable') return 'Unavailable'
  if (status === 'missing') return 'Missing'
  if (status === 'error') return 'Error'
  return String(status || 'Offline')
}

function statusPill(status) {
  if (status === 'ok') return 'pill-green'
  if (status === 'error') return 'pill-red'
  return 'pill-amber'
}

function posClass(pos) {
  const value = String(pos || '').toUpperCase()
  if (value === 'VERB') return 'pos-badge pos-verb'
  if (value === 'NOUN') return 'pos-badge pos-noun'
  if (value === 'ADJ' || value === 'ADJECTIVE') return 'pos-badge pos-adj'
  if (value === 'ADP' || value === 'ADPOSITION' || value === 'PART') return 'pos-badge pos-adp'
  return 'pos-badge pos-other'
}

function value(item) { return item || '-' }
function guardExport(event) { if (!results.value) event.preventDefault() }
async function copyCurrentJson() {
  if (!results.value) return
  await navigator.clipboard.writeText(prettyJson.value)
  copied.value = true
  window.setTimeout(() => { copied.value = false }, 1800)
}
function clear() { inputText.value = ''; results.value = null; fusionRows.value = []; error.value = ''; copied.value = false }
function loadSample() { inputText.value = 'وجدت المعلمة طالبة مجتهدة في الفصل' }

onMounted(() => {
  if (route.query.text) {
    inputText.value = String(route.query.text)
    analyze()
  }
})
</script>

<style scoped>
.analyze-hero { min-height: 250px; }
.compact-actions, .run-row { margin-top: 0; }
.run-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 16px; }
.disabled { pointer-events: none; opacity: .5; }
.copy-note { color: var(--green); font-size: 13px; font-weight: 850; }
.analysis-loading { min-height: 170px; }
.loading-stack { width: min(520px, 100%); display: grid; gap: 12px; }
.loading-stack .wide { width: 78%; }
.loading-stack .short { width: 42%; }
.tool-overview { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.tool-summary { display: grid; gap: 12px; cursor: pointer; transition: transform .16s ease, border-color .16s ease; }
.tool-summary:hover, .tool-summary.active { transform: translateY(-2px); border-color: rgba(37,99,235,.38); }
.tool-summary.unavailable { opacity: .82; background: #fafbfc; }
.tool-summary-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.tool-code { width: 36px; height: 30px; display: grid; place-items: center; border-radius: 7px; background: #e0f2fe; color: #075985; font-size: 12px; font-weight: 900; }
.tool-summary h3 { margin: 0; font-size: 17px; font-weight: 950; }
.tool-summary p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
.tool-summary strong { color: var(--navy); font-size: 18px; font-weight: 950; }
.tool-reason { color: var(--amber); font-size: 12px; font-weight: 800; line-height: 1.45; }
.analysis-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; padding: 5px; border: 1px solid var(--line); border-radius: 8px; background: #eef3f8; }
.analysis-tabs button { min-height: 38px; padding: 8px 14px; border-radius: 7px; color: var(--muted); background: transparent; cursor: pointer; font-weight: 900; }
.analysis-tabs button.active { color: var(--navy); background: #fff; box-shadow: 0 1px 7px rgba(23,32,51,.09); }
.tab-panel { display: grid; gap: 18px; }
.warning-state { padding: 18px; border: 1px solid #fcd34d; border-radius: 8px; background: #fffbeb; color: #92400e; }
.warning-state strong { display: block; font-weight: 950; }
.warning-state p { margin: 6px 0 0; }
.token-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.token-detail-card { display: grid; gap: 14px; padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; }
.token-detail-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.token-detail-head strong { font-size: 23px; font-weight: 950; }
dl { display: grid; gap: 10px; margin: 0; }
dl div { display: grid; gap: 2px; }
dt { color: var(--muted); font-size: 12px; font-weight: 900; text-transform: uppercase; }
dd { margin: 0; color: var(--ink); font-weight: 850; }
.fusion-word { font-size: 18px; font-weight: 950; }
.json-panel { max-height: 560px; overflow: auto; margin: 0; padding: 18px; border: 1px solid var(--line); border-radius: 8px; background: #101827; color: #d7e4f3; font-size: 13px; line-height: 1.65; }
@media (max-width: 1080px) { .tool-overview, .token-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 680px) { .tool-overview, .token-grid { grid-template-columns: 1fr; } }
</style>
