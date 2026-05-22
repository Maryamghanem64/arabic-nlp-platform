<template>
  <div class="page-wrap home-page page-stack">
    <section class="hero-band research-hero">
      <div class="hero-content">
        <span class="eyebrow">Arabic NLP research platform</span>
        <h1 class="hero-title">A teammate-safe Arabic NLP comparison lab.</h1>
        <p class="hero-copy">
          The platform detects installed analyzers, reports missing dependencies, and keeps
          the frontend usable even when optional tools are unavailable on another machine.
        </p>
        <div class="actions-row">
          <RouterLink class="btn btn-primary" to="/compare">Open Compare Dashboard</RouterLink>
          <RouterLink class="btn btn-ghost" to="/analyze">Deep Analysis</RouterLink>
        </div>
      </div>

      <div class="hero-metrics" aria-label="Platform highlights">
        <div v-for="stat in heroStats" :key="stat.label" class="hero-stat">
          <strong>{{ stat.value }}</strong>
          <span>{{ stat.label }}</span>
        </div>
      </div>
    </section>

    <section class="status-layout">
      <article class="panel panel-pad">
        <div class="section-head">
          <div>
            <h2 class="section-title">Live Tool Availability</h2>
            <p class="section-subtitle">Every card is driven by backend startup detection.</p>
          </div>
          <button class="btn btn-secondary" :disabled="loading" @click="fetchStatus">
            {{ loading ? 'Checking...' : 'Refresh' }}
          </button>
        </div>

        <div v-if="loading" class="tool-grid">
          <div v-for="n in 9" :key="n" class="tool-card skeleton-card">
            <span class="skeleton"></span>
            <span class="skeleton wide"></span>
            <span class="skeleton short"></span>
          </div>
        </div>
        <div v-else-if="error" class="error-state">
          <div>
            <strong>Backend unavailable</strong>
            <p>{{ error }}</p>
            <button class="btn btn-secondary" @click="fetchStatus">Retry</button>
          </div>
        </div>
        <div v-else class="tool-grid">
          <article v-for="tool in toolCards" :key="tool.key" :class="['tool-card', { unavailable: tool.status !== 'ok' }]">
            <div class="tool-card-top">
              <span class="tool-code">{{ tool.code }}</span>
              <span :class="['status-dot', statusClass(tool.status)]"></span>
            </div>
            <h3>{{ tool.name }}</h3>
            <p>{{ tool.description }}</p>
            <small v-if="tool.reason" class="tool-reason">{{ tool.reason }}</small>
            <div class="tool-card-foot">
              <span class="tool-badge">{{ tool.role }}</span>
              <span :class="['pill', statusPill(tool.status)]">{{ readableStatus(tool.status) }}</span>
            </div>
          </article>
        </div>
      </article>

      <aside class="panel panel-pad quick-panel">
        <h2 class="section-title">Quick Research Run</h2>
        <p class="section-subtitle">Load a benchmark sentence directly into the comparison dashboard.</p>
        <button class="sample-btn arabic" dir="rtl" @click="runSample">كتب الطالب الدرس</button>
        <div class="sample-notes">
          <span>Expected pattern</span>
          <strong>VERB · NOUN · NOUN</strong>
        </div>
      </aside>
    </section>

    <section class="feature-grid">
      <article v-for="feature in features" :key="feature.title" class="panel panel-pad feature-card">
        <span class="feature-index">{{ feature.index }}</span>
        <h3>{{ feature.title }}</h3>
        <p>{{ feature.text }}</p>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { getStatus } from '../api/nlpApi'

const router = useRouter()
const loading = ref(true)
const error = ref('')
const statusPayload = ref({})

const heroStats = [
  { value: '9', label: 'Tracked tools' },
  { value: '0', label: 'Crash tolerance' },
  { value: 'RTL', label: 'Arabic-first UI' },
]

const toolInfo = [
  { key: 'camel', code: 'CM', name: 'CAMeL Tools', role: 'Morphology', description: 'Disambiguation, lemma, root, POS, gender, number, tense, and confidence evidence.' },
  { key: 'farasa', code: 'FA', name: 'Farasa', role: 'Segmentation', description: 'Clitic-aware Arabic segmentation used for coverage and token evidence.' },
  { key: 'stanza', code: 'ST', name: 'Stanza', role: 'Neural syntax', description: 'Universal Dependencies POS, lemma, and dependency parsing.' },
  { key: 'qalsadi', code: 'QA', name: 'Qalsadi', role: 'Rule-based lemma', description: 'Lightweight rule-based lemmatization for lexical comparison.' },
  { key: 'arabert', code: 'AB', name: 'AraBERT', role: 'Transformer', description: 'Optional transformer-family Arabic model integration.' },
  { key: 'alkhalil', code: 'AK', name: 'AlKhalil', role: 'Java analyzer', description: 'Optional Java-based morphological analyzer.' },
  { key: 'udpipe', code: 'UD', name: 'UDPipe', role: 'UD parser', description: 'Optional UDPipe parser requiring package and Arabic model files.' },
  { key: 'madamira', code: 'MA', name: 'MADAMIRA', role: 'Java analyzer', description: 'Optional MADAMIRA integration requiring Java and local files.' },
  { key: 'sinatools', code: 'SI', name: 'SinaTools', role: 'Partner tool', description: 'Future microservice integration tracked by backend status.' },
]

const features = [
  { index: '01', title: 'Safe backend contract', text: 'Every analyzer returns status, reason, word_count, and tokens, even when unavailable.' },
  { index: '02', title: 'Partner-proof setup', text: 'Startup checks detect Java, Python packages, and model folders before a user runs analysis.' },
  { index: '03', title: 'Graceful UI fallbacks', text: 'Unavailable tools are disabled visually instead of causing undefined fields or blank tables.' },
]

const toolCards = computed(() => {
  const tools = statusPayload.value?.tools || {}
  return toolInfo.map((tool) => ({
    ...tool,
    status: tools[tool.key]?.status || 'offline',
    reason: tools[tool.key]?.reason || '',
  }))
})

function readableStatus(status) {
  if (status === 'ok') return 'Online'
  if (status === 'future_work') return 'Planned'
  if (status === 'missing_dependency') return 'Missing dependency'
  if (status === 'missing_model') return 'Missing model'
  if (status === 'missing_java') return 'Missing Java'
  if (status === 'unavailable') return 'Unavailable'
  return 'Offline'
}

function statusClass(status) {
  if (status === 'ok') return 'dot-online'
  if (status === 'future_work') return 'dot-planned'
  if (String(status).startsWith('missing') || status === 'unavailable') return 'dot-warning'
  return 'dot-offline'
}

function statusPill(status) {
  if (status === 'ok') return 'pill-green'
  if (status === 'future_work' || String(status).startsWith('missing') || status === 'unavailable') return 'pill-amber'
  return 'pill-gray'
}

async function fetchStatus() {
  loading.value = true
  error.value = ''
  try {
    statusPayload.value = await getStatus()
  } catch (e) {
    error.value = e.message || 'Cannot reach FastAPI on http://127.0.0.1:8000.'
  } finally {
    loading.value = false
  }
}

function runSample() {
  router.push({ path: '/compare', query: { text: 'كتب الطالب الدرس' } })
}

onMounted(fetchStatus)
</script>

<style scoped>
.research-hero { grid-template-columns: minmax(0, 1fr) 320px; align-items: end; }
.hero-metrics { position: relative; z-index: 1; display: grid; gap: 10px; }
.hero-stat { padding: 16px; border: 1px solid rgba(255,255,255,.2); border-radius: 8px; background: rgba(255,255,255,.11); }
.hero-stat strong, .hero-stat span { display: block; }
.hero-stat strong { font-size: 34px; line-height: 1; font-weight: 900; }
.hero-stat span { margin-top: 6px; color: rgba(255,255,255,.78); font-size: 13px; font-weight: 750; }
.status-layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 18px; }
.tool-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.tool-card { min-height: 210px; display: grid; align-content: space-between; gap: 12px; padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; }
.tool-card.unavailable { background: #fafbfc; opacity: .86; }
.tool-card-top, .tool-card-foot { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.tool-code { width: 36px; height: 30px; display: grid; place-items: center; border-radius: 7px; background: #e0f2fe; color: #075985; font-size: 12px; font-weight: 900; }
.status-dot { width: 11px; height: 11px; border-radius: 999px; }
.dot-online { background: var(--green); }
.dot-planned { background: var(--amber); }
.dot-warning { background: var(--amber); }
.dot-offline { background: var(--subtle); }
.tool-card h3, .feature-card h3 { margin: 0; font-size: 17px; font-weight: 900; }
.tool-card p, .feature-card p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.62; }
.tool-reason { color: var(--amber); font-size: 12px; font-weight: 800; line-height: 1.45; }
.skeleton-card { min-height: 150px; }
.skeleton-card .wide { width: 80%; }
.skeleton-card .short { width: 52%; }
.quick-panel { align-self: stretch; }
.sample-btn { width: 100%; margin-top: 20px; padding: 18px; border: 1px solid var(--line); border-radius: 8px; color: var(--navy); background: #f8fafc; cursor: pointer; font-size: 27px; line-height: 1.8; font-weight: 900; }
.sample-notes { display: grid; gap: 4px; margin-top: 14px; padding: 14px; border-radius: 8px; background: #f8fafc; }
.sample-notes span { color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; }
.sample-notes strong { color: var(--navy); font-size: 14px; font-weight: 900; }
.feature-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.feature-index { display: inline-grid; place-items: center; width: 38px; height: 30px; margin-bottom: 16px; border-radius: 7px; background: #e0f2fe; color: #075985; font-size: 12px; font-weight: 900; }
@media (max-width: 1020px) { .research-hero, .status-layout, .feature-grid { grid-template-columns: 1fr; } .tool-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .tool-grid { grid-template-columns: 1fr; } }
</style>
