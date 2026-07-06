<template>
  <div class="page-wrap reports-page page-stack">
    <section class="hero-band compact-hero">
      <div class="hero-content">
        <span class="eyebrow">Research report</span>
        <h1 class="hero-title">Methodology, scope, limitations, and reproducibility.</h1>
        <p class="hero-copy">A concise research-facing description of the Arabic NLP comparative platform and the claims its interface does and does not make.</p>
      </div>
    </section>

    <section class="reports-grid">
      <article v-for="section in sections" :key="section.title" class="panel panel-pad report-card">
        <span class="section-kicker">{{ section.kicker }}</span>
        <h2 class="section-title">{{ section.title }}</h2>
        <p>{{ section.text }}</p>
        <ul class="report-list"><li v-for="item in section.items" :key="item">{{ item }}</li></ul>
      </article>
    </section>

    <section class="panel panel-pad">
      <div class="section-head">
        <div>
          <h2 class="section-title">Analyzer taxonomy</h2>
          <p class="section-subtitle">Tools are grouped by research role, not by visual convenience.</p>
        </div>
      </div>
      <div class="taxonomy-grid">
        <article v-for="group in taxonomy" :key="group.name" class="taxonomy-card">
          <strong>{{ group.name }}</strong>
          <span>{{ group.tools }}</span>
          <p>{{ group.role }}</p>
        </article>
      </div>
    </section>

    <section class="panel panel-pad boundary-panel">
      <span class="section-kicker">Interpretation boundary</span>
      <h2 class="section-title">Agreement is evidence of consistency, not proof of correctness.</h2>
      <p>The platform does not infer a gold answer from analyzer consensus. Raw outputs remain inspectable, comparison is capability-scoped, and expert fusion records selected sources so decisions can be audited token by token.</p>
    </section>
  </div>
</template>

<script setup>
const sections = [
  { kicker: '01 - Scope', title: 'Comparative analyzer evidence', text: 'The system standardizes heterogeneous Arabic NLP outputs into a common inspection workflow.', items: ['Arabic input is sent to registered analyzers.', 'Outputs are normalized into comparable token fields.', 'Raw evidence remains separate from fusion and evaluation summaries.'] },
  { kicker: '02 - Alignment', title: 'Token alignment before comparison', text: 'Arabic clitics and multi-word tokenization make index-only comparison unreliable.', items: ['Aligned token evidence is inspected before feature comparison.', 'Segmentation differences are preserved as analyzer evidence.', 'Repeated surface forms are not treated as one global token identity.'] },
  { kicker: '03 - Fusion', title: 'Expert Fusion - Implemented', text: 'Feature selection is routed by analyzer strength rather than blind voting.', items: ['Farasa anchors segmentation evidence.', 'CAMeL and SinaTools contribute lexical/morphological evidence when available.', 'Stanza and UDPipe contribute UD-oriented syntax evidence.', 'The selected source, candidates, ambiguity, and conflicts remain visible.'] },
  { kicker: '04 - Evaluation', title: 'Capability-Aware Evaluation - Implemented', text: 'Unsupported fields are excluded from feature denominators.', items: ['POS Agreement is restricted to eligible POS analyzers.', 'Lemma Match, Root Agreement, and Segmentation Coverage are agreement or coverage metrics.', 'Metric contributors remain separate from capability contributors.', 'No metric is presented as gold-standard accuracy.'] },
  { kicker: '05 - Limitations', title: 'Methodological limitations', text: 'The platform is a comparative research workbench, not a gold-standard benchmark.', items: ['Analyzer conventions can differ legitimately.', 'Optional local resources affect tool participation.', 'Confidence values are analyzer/fusion signals, not calibrated correctness probabilities.', 'A labeled Arabic corpus is required for true accuracy evaluation.'] },
  { kicker: '06 - Reproducibility', title: 'Auditable run evidence', text: 'The UI preserves the data needed to inspect a run.', items: ['Participating and excluded tools are surfaced.', 'Raw JSON and exports remain available.', 'Conflict traces are shown at token/feature level.', 'Tool roles are defined centrally in frontend metadata.'] },
]

const taxonomy = [
  { name: 'Morphology / lexical evidence', tools: 'CAMeL - SinaTools - AlKhalil', role: 'Lemma, root, POS, and morphological evidence according to tool capability. SinaTools is a lazy-loaded local lexical resource.' },
  { name: 'Excluded licensed analyzer', tools: 'MADAMIRA', role: 'Excluded in the current configuration because licensed resources are missing.' },
  { name: 'Syntax', tools: 'Stanza - UDPipe', role: 'UD-oriented POS and dependency evidence.' },
  { name: 'Segmentation', tools: 'Farasa', role: 'Clitic and segment-boundary anchor.' },
  { name: 'Lexical support', tools: 'Qalsadi', role: 'Rule-based lemma and stem support.' },
  { name: 'Contextual representation', tools: 'AraBERT', role: 'Contextual transformer support only; not a lemma, root, POS, segmentation, or dependency analyzer.' },
]
</script>

<style scoped>
.compact-hero { padding: 34px 38px; }
.reports-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.report-card p,
.boundary-panel p,
.taxonomy-card p { color: var(--c-text-secondary); line-height: 1.7; }
.section-kicker { display: block; margin-bottom: 8px; color: var(--c-accent-text); font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.report-list { margin: 14px 0 0; padding-inline-start: 18px; color: var(--c-text-secondary); }
.report-list li + li { margin-top: 7px; }
.taxonomy-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.taxonomy-card { display: grid; gap: 6px; padding: 15px; border: 1px solid var(--c-border); border-radius: 10px; background: var(--c-page-bg); }
.taxonomy-card span { color: var(--c-accent-text); font-size: 13px; }
.taxonomy-card p { margin: 0; font-size: 13px; }
.boundary-panel { border-left: 4px solid var(--c-accent); }
@media (max-width: 900px) {
  .reports-grid,
  .taxonomy-grid { grid-template-columns: 1fr; }
}
</style>
