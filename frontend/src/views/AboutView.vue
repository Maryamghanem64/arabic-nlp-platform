<template>
  <div class="page-wrap about-page">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Project context</span>
      <h1 class="hero-title">About the Platform</h1>
      <p class="hero-copy">
        A practical comparative environment for Arabic NLP experiments, built with FastAPI,
        Vue, and multiple Arabic language processing toolkits.
      </p>
    </section>

    <section class="about-grid">
      <article class="panel panel-pad">
        <h2 class="section-title">Purpose</h2>
        <p class="section-subtitle">
          The platform helps researchers and students compare outputs from different Arabic NLP
          systems using the same input sentence. It focuses on morphology, segmentation,
          lemmatization, POS tagging, dependency parsing, and rule-based lexical evidence.
        </p>
      </article>

      <article class="panel panel-pad">
        <h2 class="section-title">Runtime Notes</h2>
        <p class="section-subtitle">
          Qalsadi is lightweight compared with larger model-based analyzers, so it fits the
          local FastAPI workflow well. For normal use, start the backend without
          <code>--reload</code> so NLP resources remain in memory while the server is running.
        </p>
      </article>
    </section>

    <section class="panel panel-pad tools-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">Integrated Tools</h2>
          <p class="section-subtitle">
            Every card is rendered from the shared tool config and updated from <code>GET /</code>.
          </p>
        </div>
      </div>

      <div class="tool-grid">
        <article v-for="tool in toolCards" :key="tool.key" class="tool-card">
          <div class="tool-card-head">
            <div>
              <h3>{{ tool.label }}</h3>
              <p class="tool-name-en">{{ tool.type }} · {{ tool.license }}</p>
            </div>
            <span :class="['pill', statusBadge(tool.key).className]">{{ statusBadge(tool.key).label }}</span>
          </div>

          <div class="tool-color" :style="{ backgroundColor: tool.color }"></div>

          <dl class="tool-meta">
            <div>
              <dt>Key Features</dt>
              <dd>
                <ul>
                  <li v-for="feature in tool.features" :key="feature">{{ feature }}</li>
                </ul>
              </dd>
            </div>
            <div>
              <dt>Paper</dt>
              <dd><em>{{ tool.paper }}</em></dd>
            </div>
          </dl>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { TOOL_CONFIG, TOOL_KEYS } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'

const { toolStatus } = useToolStatus()

const toolCards = computed(() => TOOL_KEYS.map((key) => ({ key, ...TOOL_CONFIG[key] })))

function statusBadge(key) {
  const status = toolStatus(key)
  if (status === 'ok') return { label: 'active', className: 'pill-green' }
  if (status === 'error') return { label: 'error', className: 'pill-red' }
  if (status === 'unavailable') return { label: 'unavailable', className: 'pill-gray' }
  if (status === 'lazy') return { label: 'loads on demand', className: 'pill-amber' }
  if (status === 'future_work') return { label: 'planned', className: 'pill-gray' }
  return { label: 'status unknown', className: 'pill-gray' }
}
</script>

<style scoped>
.compact-hero {
  padding: 34px 38px;
}

.compact-hero .hero-title {
  font-size: 38px;
}

.about-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

code {
  padding: 2px 6px;
  border-radius: 5px;
  background: #eef2f7;
  color: var(--navy);
  font-weight: 800;
}

.tools-panel {
  margin-top: 18px;
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.tool-card {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
}

.tool-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 12px;
}

.tool-card h3 {
  margin: 0;
  font-size: 18px;
}

.tool-name-en {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 13px;
  font-weight: 750;
}

.tool-color {
  width: 100%;
  height: 6px;
  margin-bottom: 14px;
  border-radius: 999px;
}

.tool-meta {
  display: grid;
  gap: 12px;
  margin: 0;
}

.tool-meta div {
  display: grid;
  gap: 4px;
}

dt {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

dd {
  margin: 0;
  color: var(--ink);
  font-weight: 800;
}

ul {
  margin: 0;
  padding-inline-start: 18px;
}

li {
  margin: 2px 0;
}

@media (max-width: 980px) {
  .about-grid,
  .tool-grid {
    grid-template-columns: 1fr;
  }
}
</style>
