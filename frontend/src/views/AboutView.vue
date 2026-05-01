<template>
  <div class="container">
    <h1 class="page-title">About This Platform</h1>

    <div class="card">
      <h2>Project Overview</h2>
      <p>
        This platform compares many state-of-the-art Arabic NLP tools for morphological analysis,
        segmentation, and dependency parsing. Built as a graduation project to evaluate and
        visualize differences in Arabic language processing systems.
      </p>
    </div>

    <h2 class="section-title">Tools</h2>

    <div class="card tool-info">
      <div class="tool-header">
        <span class="badge badge-blue tool-badge">CAMeL Tools</span>
        <span class="year">2020–2024  ·  MIT License</span>
      </div>
      <p>Comprehensive Arabic NLP toolkit developed by NYU Abu Dhabi. Provides morphological
      analysis, disambiguation, lemmatization, POS tagging, and diacritization for MSA and
      multiple Arabic dialects.</p>
      <div class="metrics">
        <span class="metric">MSA POS: 98.9%</span>
        <span class="metric">Lemma: 96.3%</span>
        <span class="metric">4 Arabic varieties</span>
      </div>
    </div>

    <div class="card tool-info">
      <div class="tool-header">
        <span class="badge badge-purple tool-badge">Farasa</span>
        <span class="year">2016  ·  Free (research)</span>
      </div>
      <p>Fast Arabic segmentation and NLP pipeline developed by QCRI. Uses SVM-rank classifier
      trained on Penn Arabic Treebank. Provides segmentation, POS tagging, lemmatization,
      NER, and dependency parsing.</p>
      <div class="metrics">
        <span class="metric">Segmentation: 98.94%</span>
        <span class="metric">Word Seg F1: 89%</span>
      </div>
    </div>

    <div class="card tool-info">
      <div class="tool-header">
        <span class="badge badge-green tool-badge">Stanza</span>
        <span class="year">2020  ·  Apache 2.0</span>
      </div>
      <p>Stanford NLP neural pipeline supporting 66 languages. Uses BiLSTM and seq2seq models
      trained on Universal Dependencies treebanks. Provides full UD pipeline including
      dependency parsing.</p>
      <div class="metrics">
        <span class="metric">UPOS: 94.89%</span>
        <span class="metric">LAS: 79.33%</span>
        <span class="metric">NER F1: 74.3%</span>
      </div>
    </div>

    <div class="card">
      <h2>Reference</h2>
      <p class="ref">
        Saadiyeh et al., "A Comparative Study of Arabic Morphological Analyzers,"
        <em>IAES International Journal of Artificial Intelligence (IJ-AI)</em>.
      </p>
    </div>
  </div>
</template>

<style scoped>
.page-title { font-size: 1.5rem; color: #1F3864; margin-bottom: 20px; }
.section-title { font-size: 1.2rem; color: #1F3864; margin: 24px 0 12px; }
.card h2 { font-size: 1.1rem; color: #1F3864; margin-bottom: 10px; }
.card p { color: #5D6D7E; line-height: 1.6; }
.tool-info { margin-bottom: 16px; }
.tool-header { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.tool-badge { font-size: 0.9rem !important; padding: 6px 14px !important; }
.year { color: #5D6D7E; font-size: 0.85rem; }
.metrics { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.metric { background: #f8fafc; border: 1px solid #e2e8f0; padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; color: #1F3864; font-weight: 600; }
.ref { font-size: 0.95rem; }
</style>