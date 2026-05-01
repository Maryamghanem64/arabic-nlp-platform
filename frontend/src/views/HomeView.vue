<template>
  <div class="container">
    <!-- Hero -->
    <div class="hero card">
      <h1>Arabic NLP Comparative Platform</h1>
      <p class="subtitle">
        Compare state-of-the-art Arabic morphological analysis tools:
        CAMeL Tools, Farasa, and Stanza — side by side.
      </p>
      <div class="hero-actions">
        <RouterLink to="/analyze" class="btn btn-primary">Start Analyzing →</RouterLink>
        <RouterLink to="/compare" class="btn btn-outline">Compare Tools</RouterLink>
      </div>
    </div>

    <!-- Tools Status -->
    <h2 class="section-title">Tools Status</h2>
    <div v-if="loading" class="loading"><div class="spinner"></div> Loading...</div>
    <div v-else-if="error" class="error-msg">{{ error }}</div>
    <div v-else class="tools-grid">
      <div v-for="(info, name) in tools" :key="name" class="tool-card card">
        <div class="tool-header">
          <span class="tool-name">{{ name.toUpperCase() }}</span>
          <span :class="info.status === 'ok' ? 'badge badge-green' : 'badge badge-red'">
            {{ info.status === 'ok' ? '● Online' : '● Offline' }}
          </span>
        </div>
        <div class="capabilities">
          <span v-for="cap in info.capabilities" :key="cap" class="badge badge-blue cap">
            {{ cap }}
          </span>
        </div>
      </div>
    </div>

    <!-- Demo Section -->
    <div class="card demo-card">
      <h2 class="demo-title">Try It Out</h2>
      <p class="demo-description">Click the example sentence below to analyze it:</p>
      <button class="demo-sentence arabic" @click="runDemo">
        {{ exampleSentence }}
      </button>
    </div>

    <!-- Features -->
    <h2 class="section-title">What This Platform Does</h2>
    <div class="features-grid">
      <div class="feature card">
        <div class="feature-icon">🔍</div>
        <h3>Morphological Analysis</h3>
        <p>Extract lemma, root, POS, gender, number, tense and more from Arabic text.</p>
      </div>
      <div class="feature card">
        <div class="feature-icon">✂️</div>
        <h3>Segmentation</h3>
        <p>Split Arabic words into their morphological components using Farasa.</p>
      </div>
      <div class="feature card">
        <div class="feature-icon">🌳</div>
        <h3>Dependency Parsing</h3>
        <p>Understand sentence structure — subject, object, and grammatical relations.</p>
      </div>
      <div class="feature card">
        <div class="feature-icon">📊</div>
        <h3>Tool Comparison</h3>
        <p>See how different NLP tools analyze the same Arabic sentence.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const loading = ref(true)
const error = ref('')
const tools = ref({})
const exampleSentence = 'الطلاب يقرؤون الكتب في المكتبة'

onMounted(async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8000/')
    tools.value = res.data.tools
  } catch (e) {
    error.value = 'Failed to connect to backend. Make sure the server is running.'
    console.error(e)
  } finally {
    loading.value = false
  }
})

function runDemo() {
  router.push({ 
    path: '/analyze', 
    query: { text: exampleSentence }
  })
}
</script>

<style scoped>
.hero {
  text-align: center;
  padding: 48px 32px;
  background: linear-gradient(135deg, #1F3864 0%, #2E5FA3 100%);
  color: white;
  margin-bottom: 32px;
}

.hero h1 {
  font-size: 2rem;
  margin-bottom: 12px;
}

.subtitle {
  font-size: 1.05rem;
  color: #BDD7EE;
  max-width: 600px;
  margin: 0 auto 24px;
  line-height: 1.6;
}

.hero-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.btn-outline {
  padding: 10px 24px;
  border: 2px solid white;
  border-radius: 8px;
  color: white;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-outline:hover {
  background: white;
  color: #1F3864;
}

.section-title {
  font-size: 1.3rem;
  color: #1F3864;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid #BDD7EE;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.tool-card { padding: 20px; }

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.tool-name {
  font-size: 1.1rem;
  font-weight: 700;
  color: #1F3864;
}

.capabilities { display: flex; flex-wrap: wrap; gap: 6px; }

.cap { font-size: 0.75rem !important; }

.demo-card {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2f6 100%);
  margin-bottom: 32px;
}

.demo-title {
  font-size: 1.2rem;
  color: #1F3864;
  margin-bottom: 8px;
}

.demo-description {
  color: #5D6D7E;
  margin-bottom: 16px;
}

.demo-sentence {
  display: block;
  width: 100%;
  padding: 16px 20px;
  font-size: 1.3rem;
  background: white;
  border: 2px solid #2E5FA3;
  border-radius: 8px;
  color: #1F3864;
  cursor: pointer;
  text-align: center;
  transition: all 0.2s;
}

.demo-sentence:hover {
  background: #2E5FA3;
  color: white;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.feature { text-align: center; padding: 24px 16px; }

.feature-icon { font-size: 2rem; margin-bottom: 12px; }

.feature h3 { font-size: 1rem; color: #1F3864; margin-bottom: 8px; }

.feature p { font-size: 0.85rem; color: #5D6D7E; line-height: 1.5; }
</style>
