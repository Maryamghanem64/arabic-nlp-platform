<template>
  <div class="page-wrap home-page">
    <section class="hero-band">
      <span class="eyebrow">Research interface</span>
      <h1 class="hero-title">Compare Arabic NLP tools with one clean workflow.</h1>
      <p class="hero-copy">
        Run Arabic text through CAMeL Tools, Farasa, Stanza, and Qalsadi, then inspect
        morphology, segmentation, dependencies, rule-based lemmas, and tool agreement.
      </p>
      <div class="actions-row">
        <RouterLink class="btn btn-primary" to="/analyze">Analyze Text</RouterLink>
        <RouterLink class="btn btn-ghost" to="/compare">Compare Tools</RouterLink>
      </div>
    </section>

    <section class="status-layout">
      <div class="panel panel-pad">
        <div class="section-head">
          <div>
            <h2 class="section-title">Backend Status</h2>
            <p class="section-subtitle">Live availability reported by the FastAPI service.</p>
          </div>
          <button class="btn btn-secondary" @click="fetchStatus" :disabled="loading">
            Refresh
          </button>
        </div>

        <div v-if="loading" class="loading-state">Checking the backend...</div>
        <div v-else-if="error" class="error-state">{{ error }}</div>
        <div v-else class="tool-grid">
          <article v-for="tool in toolCards" :key="tool.key" class="tool-card">
            <div class="tool-meta">
              <span class="tool-dot" :class="statusClass(tool.status)"></span>
              <div>
                <h3>{{ tool.name }}</h3>
                <p>{{ tool.description }}</p>
              </div>
            </div>
            <span class="pill" :class="tool.status === 'ok' ? 'pill-green' : 'pill-red'">
              {{ readableStatus(tool.status) }}
            </span>
          </article>
        </div>
      </div>

      <aside class="panel panel-pad quick-panel">
        <h2 class="section-title">Quick Run</h2>
        <p class="section-subtitle">
          Use the sample sentence to test the full pipeline.
        </p>
        <button class="sample-btn arabic" dir="rtl" @click="runSample">
          ذهب محمد إلى جامعة بيرزيت لدراسة اللغة العربية
        </button>
      </aside>
    </section>

    <section class="feature-grid">
      <article v-for="feature in features" :key="feature.title" class="panel panel-pad feature-card">
        <span class="feature-code">{{ feature.code }}</span>
        <h3>{{ feature.title }}</h3>
        <p>{{ feature.text }}</p>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const router = useRouter()
const loading = ref(true)
const error = ref('')
const tools = ref({})

const toolInfo = [
  { key: 'camel', name: 'CAMeL Tools', description: 'Morphology, lemma, root, POS, and confidence.' },
  { key: 'farasa', name: 'Farasa', description: 'Arabic segmentation and clitic splitting.' },
  { key: 'stanza', name: 'Stanza', description: 'Universal POS, lemma, and dependency parsing.' },
  { key: 'qalsadi', name: 'Qalsadi', description: 'Rule-based Arabic lemmatization and POS evidence.' },
  { key: 'sinatools', name: 'SinaTools', description: 'Microservice planned (excluded from main app).' },
]

const features = [
  { code: '01', title: 'Unified Analysis', text: 'One input sends text to every configured NLP engine and returns consistent JSON.' },
  { code: '02', title: 'Token-Level Evidence', text: 'Inspect lemma, root, POS, features, segmentation, dependencies, and Qalsadi stems per word.' },
  { code: '03', title: 'Agreement View', text: 'Spot where tools agree or disagree and use the fusion output as a research summary.' },
]

const toolCards = computed(() =>
  toolInfo.map((tool) => ({
    ...tool,
    status: tools.value?.[tool.key]?.status || 'offline',
  })),
)

function readableStatus(status) {
  if (status === 'ok') return 'Online'
  if (status === 'future_work') return 'Future work'
  return 'Offline'
}

function statusClass(status) {
  if (status === 'ok') return 'dot-online'
  if (status === 'future_work') return 'dot-offline'
  return 'dot-offline'
}

async function fetchStatus() {
  loading.value = true
  error.value = ''

  try {
    const { data } = await axios.get(`${API}/`, { timeout: 2500 })
    tools.value = data.tools || {}
  } catch {
    error.value = 'Cannot reach the backend. Start FastAPI on http://127.0.0.1:8000.'
  } finally {
    loading.value = false
  }
}

function runSample() {
  router.push({
    path: '/analyze',
    query: { text: 'ذهب محمد إلى جامعة بيرزيت لدراسة اللغة العربية' },
  })
}

onMounted(fetchStatus)
</script>

<style scoped>
.status-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 18px;
  margin-top: 18px;
}

.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 18px;
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.tool-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.tool-meta {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.tool-meta h3,
.feature-card h3 {
  margin: 0;
  font-size: 16px;
}

.tool-meta p,
.feature-card p {
  margin: 6px 0 0;
  color: var(--muted);
  line-height: 1.55;
  font-size: 14px;
}

.tool-dot {
  width: 11px;
  height: 11px;
  margin-top: 5px;
  border-radius: 50%;
  flex: 0 0 auto;
}

.dot-online {
  background: var(--green);
}

.dot-offline {
  background: var(--red);
}

.quick-panel {
  align-self: stretch;
}

.sample-btn {
  width: 100%;
  margin-top: 18px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f8fafc;
  color: var(--navy);
  cursor: pointer;
  font-size: 22px;
  line-height: 1.8;
  font-weight: 800;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

.feature-code {
  display: inline-grid;
  place-items: center;
  width: 36px;
  height: 28px;
  margin-bottom: 18px;
  border-radius: 7px;
  background: #e0f2fe;
  color: #075985;
  font-size: 12px;
  font-weight: 900;
}

@media (max-width: 940px) {
  .status-layout,
  .feature-grid {
    grid-template-columns: 1fr;
  }

  .tool-grid {
    grid-template-columns: 1fr;
  }
}
</style>
