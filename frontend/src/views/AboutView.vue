<template>
  <div class="page-wrap reports-page page-stack">
    <section class="hero-band compact-hero">
      <span class="eyebrow">Project report</span>
      <h1 class="hero-title">Graduation project review and improvement roadmap.</h1>
      <p class="hero-copy">
        This page summarizes the platform from a research, software engineering, and UX perspective,
        and highlights the changes that make the project read like a real comparative Arabic NLP system.
      </p>
    </section>

    <section class="reports-grid">
      <article class="panel panel-pad report-card">
        <h2 class="section-title">Strengths</h2>
        <ul class="report-list">
          <li>Multiple Arabic analyzers are integrated behind a unified frontend.</li>
          <li>Token-level comparison, fusion, and evaluation are already available.</li>
          <li>The backend exposes structured outputs, confidence, and conflict traces.</li>
          <li>The project already has the bones of a research workflow, not just a demo UI.</li>
        </ul>
      </article>

      <article class="panel panel-pad report-card">
        <h2 class="section-title">Weaknesses</h2>
        <ul class="report-list">
          <li>The old interface mixed Arabic and English and did not feel like one product.</li>
          <li>The dashboard lacked clear scientific summary cards and benchmark context.</li>
          <li>Visualization depth was limited for a comparative research platform.</li>
          <li>Some pages read like raw analysis tables rather than curated research views.</li>
        </ul>
      </article>
    </section>

    <section class="panel panel-pad report-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">Priority Improvements</h2>
          <p class="section-subtitle">These are the most important changes for graduation readiness.</p>
        </div>
      </div>

      <div class="priority-grid">
        <article v-for="item in priorities" :key="item.title" class="priority-card">
          <span class="priority-index">{{ item.index }}</span>
          <h3>{{ item.title }}</h3>
          <p>{{ item.text }}</p>
        </article>
      </div>
    </section>

    <section class="panel panel-pad report-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">Files to Review</h2>
          <p class="section-subtitle">These are the highest-value frontend files for continued refinement.</p>
        </div>
      </div>

      <div class="file-grid">
        <a class="file-card" href="/frontend/src/App.vue">App.vue</a>
        <a class="file-card" href="/frontend/src/views/HomeView.vue">HomeView.vue</a>
        <a class="file-card" href="/frontend/src/views/AnalyzeView.vue">AnalyzeView.vue</a>
        <a class="file-card" href="/frontend/src/views/CompareView.vue">CompareView.vue</a>
        <a class="file-card" href="/frontend/src/views/EvaluateView.vue">EvaluateView.vue</a>
        <a class="file-card" href="/frontend/src/views/SmartAnalysisView.vue">SmartAnalysisView.vue</a>
      </div>
    </section>
  </div>
</template>

<script setup>
const priorities = [
  {
    index: '01',
    title: 'Polish the dashboard narrative',
    text: 'Keep the landing page focused on live status, benchmark metrics, and system readiness so it immediately feels research-oriented.',
  },
  {
    index: '02',
    title: 'Strengthen scientific visualization',
    text: 'Add more charts for agreement, confidence, conflicts, and tool ranking so the platform can support a thesis defense convincingly.',
  },
  {
    index: '03',
    title: 'Keep Arabic only where it matters',
    text: 'Reserve Arabic for user input, token output, and linguistic examples while keeping all UI chrome in English.',
  },
  {
    index: '04',
    title: 'Preserve graceful fallback behavior',
    text: 'Unavailable services should degrade to clear status cards instead of breaking the workflow or hiding information.',
  },
]
</script>

<style scoped>
.compact-hero {
  padding: 34px 38px;
}

.reports-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.report-list {
  margin: 0;
  padding-inline-start: 18px;
  color: var(--c-text-secondary);
  line-height: 1.7;
}

.report-list li + li {
  margin-top: 8px;
}

.priority-grid,
.file-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.priority-card,
.file-card {
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 14px;
  background: var(--c-page-bg);
}

.priority-index {
  display: inline-grid;
  place-items: center;
  width: 36px;
  height: 28px;
  margin-bottom: 12px;
  border-radius: 8px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 12px;
  font-weight: 700;
}

.priority-card h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
}

.priority-card p,
.file-card {
  color: var(--c-text-secondary);
}

.file-card {
  display: block;
  text-decoration: none;
  font-weight: 600;
}

.file-card:hover {
  border-color: var(--c-accent-border);
  color: var(--c-accent-text);
}

@media (max-width: 980px) {
  .reports-grid,
  .priority-grid,
  .file-grid {
    grid-template-columns: 1fr;
  }
}
</style>
