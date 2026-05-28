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
              <th>AraBERT</th>
              <th>UDPipe</th>
              <th>AlKhalil</th>
              <th>MADAMIRA</th>
              <th>SinaTools</th>
            </tr>
          </thead>

          <tbody>
            <tr v-for="row in featureRows" :key="row.feature">
              <td>{{ row.feature }}</td>
              <td>{{ mark(row.camel) }}</td>
              <td>{{ mark(row.farasa) }}</td>
              <td>{{ mark(row.stanza) }}</td>
              <td>{{ mark(row.qalsadi) }}</td>
              <td>{{ mark(row.arabert) }}</td>
              <td>{{ mark(row.udpipe) }}</td>
              <td>{{ mark(row.alkhalil) }}</td>
              <td>{{ mark(row.madamira) }}</td>
              <td>{{ mark(row.sinatools) }}</td>
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
    description:
      'Arabic morphological disambiguation, root, lemma, POS, gender, number, and tense.',
  },
  {
    name: 'Farasa',
    type: 'Segmentation',
    pill: 'pill pill-violet',
    description:
      'Fast token segmentation and clitic splitting for Arabic text.',
  },
  {
    name: 'Stanza',
    type: 'Syntax',
    pill: 'pill pill-green',
    description:
      'Universal Dependencies pipeline with POS, lemma, dependency parsing, and neural linguistic analysis.',
  },
  {
    name: 'Qalsadi',
    type: 'Rule-based',
    pill: 'pill pill-amber',
    description:
      'Rule-based Arabic lemmatizer providing stems, lemmas, POS tags, and lexical evidence.',
  },

  {
    name: 'AraBERT',
    type: 'Transformer',
    pill: 'pill pill-rose',
    description:
      'Transformer-based Arabic language model for contextual embeddings and semantic analysis.',
  },
  {
    name: 'UDPipe',
    type: 'UD Parser',
    pill: 'pill pill-cyan',
    description:
      'Universal Dependencies parser for tokenization, POS tagging, and syntactic dependency analysis.',
  },
  {
    name: 'AlKhalil',
    type: 'Java Analyzer',
    pill: 'pill pill-orange',
    description:
      'Classical Arabic morphological analyzer based on roots and patterns.',
  },
  {
    name: 'MADAMIRA',
    type: 'Enterprise NLP',
    pill: 'pill pill-red',
    description:
      'Hybrid Arabic NLP system combining morphological analysis, disambiguation, and POS tagging.',
  },
  {
    name: 'SinaTools',
    type: 'Research AI',
    pill: 'pill pill-slate',
    description:
      'Large-scale Arabic NLP toolkit and future microservice integration for advanced processing.',
  },
]

const featureRows = [
  { feature: 'Lemma', camel: true, farasa: false, stanza: true, qalsadi: true, arabert: false, udpipe: true, alkhalil: true, madamira: true, sinatools: true },
  { feature: 'Stem', camel: false, farasa: false, stanza: false, qalsadi: true, arabert: false, udpipe: false, alkhalil: true, madamira: true, sinatools: false },
  { feature: 'Root', camel: true, farasa: false, stanza: false, qalsadi: false, arabert: false, udpipe: false, alkhalil: true, madamira: true, sinatools: false },
  { feature: 'POS', camel: true, farasa: false, stanza: true, qalsadi: true, arabert: true, udpipe: true, alkhalil: true, madamira: true, sinatools: true },
  { feature: 'Gender', camel: true, farasa: false, stanza: true, qalsadi: false, arabert: false, udpipe: true, alkhalil: true, madamira: true, sinatools: false },
  { feature: 'Number', camel: true, farasa: false, stanza: true, qalsadi: false, arabert: false, udpipe: true, alkhalil: true, madamira: true, sinatools: false },
  { feature: 'Segmentation', camel: false, farasa: true, stanza: false, qalsadi: false, arabert: false, udpipe: true, alkhalil: false, madamira: true, sinatools: true },
  { feature: 'Dependency', camel: false, farasa: false, stanza: true, qalsadi: false, arabert: false, udpipe: true, alkhalil: false, madamira: false, sinatools: false },
  { feature: 'Transformer Embeddings', camel: false, farasa: false, stanza: false, qalsadi: false, arabert: true, udpipe: false, alkhalil: false, madamira: false, sinatools: true },
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
  overflow-x: auto;
}

.matrix-table table {
  min-width: 1200px;
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

/* Pills */
.pill-rose { background: #ffe4ef; color: #be185d; }
.pill-cyan { background: #dff7ff; color: #0f766e; }
.pill-orange { background: #fff1df; color: #c2410c; }
.pill-red { background: #ffe2e2; color: #b91c1c; }
.pill-slate { background: #e5e7eb; color: #334155; }
</style>