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
        <h2 class="section-title">Integrated Tools</h2>
        <p class="section-subtitle">Each tool contributes a different kind of linguistic evidence.</p>
      </div>

      <div class="tool-grid">
        <article v-for="tool in tools" :key="tool.name" class="tool-card">
          <span :class="tool.pill">{{ tool.type }}</span>
          <h3>{{ tool.name }}</h3>
          <p>{{ tool.description }}</p>
        </article>
      </div>
    </section>

    <section class="panel panel-pad">
      <h2 class="section-title">Feature Matrix</h2>
      <div class="table-scroll matrix-table">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>CAMeL</th>
              <th>Farasa</th>
              <th>Stanza</th>
              <th>Qalsadi</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in featureRows" :key="row.feature">
              <td>{{ row.feature }}</td>
              <td>{{ mark(row.camel) }}</td>
              <td>{{ mark(row.farasa) }}</td>
              <td>{{ mark(row.stanza) }}</td>
              <td>{{ mark(row.qalsadi) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
const tools = [
  {
    name: 'CAMeL Tools',
    type: 'Morphology',
    pill: 'pill pill-blue',
    description: 'Arabic morphological disambiguation, root, lemma, POS, gender, number, and tense.',
  },
  {
    name: 'Farasa',
    type: 'Segmentation',
    pill: 'pill pill-violet',
    description: 'Fast token segmentation and clitic splitting for Arabic text.',
  },
  {
    name: 'Stanza',
    type: 'Syntax',
    pill: 'pill pill-green',
    description: 'Universal Dependencies pipeline with POS, lemma, and dependency relations.',
  },
  {
    name: 'Qalsadi',
    type: 'Rule-based',
    pill: 'pill pill-amber',
    description: 'Rule-based Arabic lemmatizer that provides lemmas, stems, unvocalized forms, and Arabic POS tags.',
  },
]

const featureRows = [
  { feature: 'Lemma', camel: true, farasa: false, stanza: true, qalsadi: true },
  { feature: 'Stem', camel: false, farasa: false, stanza: false, qalsadi: true },
  { feature: 'Root', camel: true, farasa: false, stanza: false, qalsadi: false },
  { feature: 'POS', camel: true, farasa: false, stanza: true, qalsadi: true },
  { feature: 'Gender', camel: true, farasa: false, stanza: true, qalsadi: false },
  { feature: 'Number', camel: true, farasa: false, stanza: true, qalsadi: false },
  { feature: 'Segmentation', camel: false, farasa: true, stanza: false, qalsadi: false },
  { feature: 'Dependency', camel: false, farasa: false, stanza: true, qalsadi: false },
]

function mark(value) {
  return value ? 'Yes' : '-'
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

.section-head {
  margin-bottom: 18px;
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.tool-card {
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.tool-card h3 {
  margin: 16px 0 8px;
  font-size: 16px;
}

.tool-card p {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}

.matrix-table {
  margin-top: 18px;
}

.matrix-table td:not(:first-child),
.matrix-table th:not(:first-child) {
  text-align: center;
}

@media (max-width: 980px) {
  .about-grid,
  .tool-grid {
    grid-template-columns: 1fr;
  }
}
</style>
