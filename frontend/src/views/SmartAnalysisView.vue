<template>
  <div class="page-wrap smart-page page-stack">
    <section class="hero-band smart-hero">
      <div class="hero-content">
        <span class="eyebrow">Expert fusion laboratory</span>
        <h1 class="hero-title">Capability-based expert fusion, explained token by token.</h1>
        <p class="hero-copy">
          This view routes each linguistic feature to eligible analyzers, preserves the selected source,
          and exposes supporting evidence, confidence, and conflicts for token-level inspection.
        </p>
      </div>
      <div class="tool-legend" aria-label="Tool color legend">
        <ToolBadge v-for="(_, tool) in visibleToolColors" :key="tool" :tool="tool" />
      </div>
    </section>

    <section class="panel panel-pad input-section">
      <div class="section-head">
        <div>
          <h2 class="section-title">Arabic Input</h2>
          <p class="section-subtitle">Arabic stays in the input and the token outputs; the interface remains English.</p>
        </div>
      </div>

      <div class="input-row">
        <textarea
          id="smart-input"
          v-model="inputText"
          class="arabic-input"
          placeholder="Example: قرأ الطالب الكتب في المكتبة"
          rows="2"
          dir="rtl"
          lang="ar"
        ></textarea>
        <button class="analyze-btn" :disabled="loading || !inputText.trim()" @click="runSmartAnalysis">
          <span v-if="loading" class="spinner" aria-hidden="true"></span>
          <span v-else>Run fusion</span>
        </button>
      </div>

      <div class="examples-row">
        <span class="examples-label">Quick examples:</span>
        <button v-for="ex in EXAMPLE_SENTENCES" :key="ex" class="example-chip" @click="runExample(ex)">
          {{ ex }}
        </button>
      </div>
    </section>

    <div v-if="error" class="error-banner">{{ error }}</div>

    <div v-if="loading" class="loading-state smart-loading">
      <div class="loading-spinner-large"></div>
      <p>Running {{ activeToolCount }} tools and collecting fused evidence...</p>
    </div>

    <template v-if="fusionResult && !loading">
      <section class="summary-bar" aria-label="Fusion summary">
        <div class="summary-item">
          <span class="summary-label">Tokens</span>
          <strong>{{ fusionResult.length }}</strong>
        </div>
        <div class="summary-item">
          <span class="summary-label">Participating tools</span>
          <strong class="tool-list">{{ activeToolsLabel }}</strong>
        </div>
        <div class="summary-item">
          <span class="summary-label">Conflicts</span>
          <strong class="conflict-count">{{ totalConflicts }}</strong>
        </div>
        <div class="summary-item">
          <span class="summary-label">Average confidence</span>
          <strong>{{ averageConfidenceLabel }}</strong>
        </div>
      </section>

      <section class="fusion-participation-grid" aria-label="Fusion tool participation">
        <article class="participation-card">
          <span class="summary-label">Active evidence tools</span>
          <strong>{{ activeTools.length ? activeTools.join(' / ') : 'Not reported' }}</strong>
        </article>
        <article class="participation-card">
          <span class="summary-label">Degraded tools</span>
          <strong>{{ degradedTools.length ? degradedTools.join(' / ') : 'None reported' }}</strong>
        </article>
        <article class="participation-card">
          <span class="summary-label">Excluded tools</span>
          <strong>{{ excludedTools.length ? excludedTools.join(' / ') : 'None reported' }}</strong>
        </article>
      </section>

      <section class="fusion-method-grid">
        <article class="panel panel-pad decision-policy">
          <span class="feature-label">Decision policy</span>
          <h2>Capability first, evidence second.</h2>
          <p>Fusion does not use blind majority voting. Each feature is routed to eligible experts, aligned candidate evidence is inspected, and the selected source is preserved in the token trace.</p>
          <div class="role-grid">
            <div v-for="role in expertRoles" :key="role.role" class="role-card">
              <strong>{{ role.role }}</strong><span>{{ role.tools }}</span><small>{{ role.features }}</small>
            </div>
          </div>
        </article>
        <ScientificChart
          type="line"
          title="Fusion Confidence by Token"
          subtitle="Reported confidence for each fused token; confidence is not a correctness score."
          badge="Token trace"
          :labels="confidenceTimeline.labels"
          :datasets="confidenceTimeline.datasets"
          :height="300"
          aria-label="Fusion confidence by token"
          empty-title="No confidence trace"
          empty-text="The fusion payload did not include token confidence scores."
        />
      </section>

      <section class="tokens-grid" aria-label="Token fusion cards">
        <article
          v-for="(token, idx) in fusionResult"
          :key="`${token.word}-${idx}`"
          class="token-card"
          :class="{ 'has-conflicts': token.conflicts.length }"
        >
          <header class="token-header">
            <div>
              <div class="token-word arabic-word">{{ token.word }}</div>
              <div class="token-meta">Selected source map and conflict trace</div>
            </div>
            <ConfidenceBadge :level="token.final.confidence_level" :score="token.final.confidence_score" />
          </header>

          <div class="features-grid">
            <div class="feature-item">
              <div class="feature-label-row">
                <span class="feature-label">Lemma</span>
                <ToolBadge :tool="sourceFor(token, 'lemma', 'camel')" />
              </div>
              <span v-if="hasValue(token.final.lemma)" class="feature-value arabic-value">{{ token.final.lemma }}</span>
              <EmptyCell v-else />
              <small class="source-line">{{ primarySourceLabel(token, 'lemma') }}</small>
            </div>

            <div class="feature-item">
              <div class="feature-label-row">
                <span class="feature-label">Root</span>
                <ToolBadge :tool="sourceFor(token, 'root', 'camel')" />
              </div>
              <template v-if="hasValue(token.final.root)">
                <span class="feature-value root-value ltr-value">{{ token.final.root }}</span>
                <span v-if="token.final.root_type" class="root-type-badge">{{ rootTypeLabel(token.final.root_type) }}</span>
              </template>
              <EmptyCell v-else />
              <small class="source-line">{{ primarySourceLabel(token, 'root') }}</small>
            </div>

            <div class="feature-item">
              <div class="feature-label-row">
                <span class="feature-label">POS</span>
                <ToolBadge :tool="sourceFor(token, 'pos', 'camel')" />
              </div>
              <span v-if="hasValue(token.final.pos)" class="pos-badge ltr-value">
                {{ POS_AR[token.final.pos] || token.final.pos }}
              </span>
              <EmptyCell v-else />
              <small class="source-line">{{ primarySourceLabel(token, 'pos') }}</small>
            </div>

            <div v-if="hasValue(token.final.gloss)" class="feature-item">
              <div class="feature-label-row">
                <span class="feature-label">English gloss</span>
                <ToolBadge :tool="sourceFor(token, 'gloss', 'camel')" />
              </div>
              <span class="gloss-value ltr-value">{{ token.final.gloss }}</span>
            </div>

            <div v-if="token.final.gender || token.final.number" class="feature-item">
              <span class="feature-label">Gender and number</span>
              <div class="morph-pills">
                <span v-if="token.final.gender" class="morph-pill">{{ genderLabel(token.final.gender) }}</span>
                <span v-if="token.final.number" class="morph-pill">{{ NUMBER_AR[token.final.number] || token.final.number }}</span>
              </div>
            </div>

            <div v-if="token.final.tense" class="feature-item">
              <span class="feature-label">Tense</span>
              <span class="morph-pill">{{ TENSE_AR[token.final.tense] || token.final.tense }}</span>
            </div>

            <div v-if="token.final.case || token.final.definite || token.final.voice" class="feature-item">
              <div class="feature-label-row">
                <span class="feature-label">Case and definiteness</span>
                <ToolBadge :tool="sourceFor(token, 'case', 'stanza')" />
              </div>
              <div class="morph-pills">
                <span v-if="token.final.case" class="morph-pill stanza-pill">{{ CASE_AR[token.final.case] || token.final.case }}</span>
                <span v-if="token.final.definite" class="morph-pill stanza-pill">{{ definiteLabel(token.final.definite) }}</span>
                <span v-if="token.final.voice" class="morph-pill stanza-pill">{{ voiceLabel(token.final.voice) }}</span>
              </div>
            </div>

            <div v-if="segmentationList(token).length" class="feature-item full-width">
              <div class="feature-label-row">
                <span class="feature-label">Segmentation</span>
                <ToolBadge :tool="sourceFor(token, 'segmentation', 'farasa')" />
              </div>
              <div class="seg-pills">
                <span v-for="(seg, si) in segmentationList(token)" :key="`${idx}-${si}`" class="seg-pill arabic-value">{{ seg }}</span>
              </div>
            </div>

            <div v-if="dependencyLabel(token.final.dependency)" class="feature-item full-width">
              <div class="feature-label-row">
                <span class="feature-label">Dependency</span>
                <ToolBadge :tool="sourceFor(token, 'dependency', 'stanza')" />
              </div>
              <div class="dep-row">
                <span class="dep-rel">{{ dependencyLabel(token.final.dependency) }}</span>
                <span v-if="token.final.dependency?.head_text" class="dep-head arabic-value">{{ token.final.dependency.head_text }}</span>
              </div>
            </div>
          </div>

          <section class="candidate-evidence-panel">
            <div class="candidate-panel-head">
              <div>
                <span class="feature-label">Eligible analyzer evidence</span>
                <p>Returned candidate values from feature-eligible tools. These values are evidence, not votes.</p>
              </div>
              <span class="candidate-note">selected source stays backend-defined</span>
            </div>

            <div class="candidate-feature-grid">
              <article
                v-for="feature in candidateFeatures"
                :key="`${idx}-${feature}`"
                class="candidate-feature-card"
              >
                <header>
                  <strong>{{ featureLabel(feature) }}</strong>
                  <ToolBadge :tool="sourceFor(token, feature, '')" />
                </header>

                <div v-if="candidateEvidence(idx, feature).length" class="candidate-list">
                  <div
                    v-for="item in candidateEvidence(idx, feature)"
                    :key="`${idx}-${feature}-${item.tool}`"
                    class="candidate-row"
                    :class="{ selected: sourceFor(token, feature, '') === item.tool }"
                  >
                    <ToolBadge :tool="item.tool" />
                    <span
                      class="candidate-value"
                      :class="{ arabic: ['lemma', 'root', 'segmentation'].includes(feature) }"
                      :dir="['lemma', 'root', 'segmentation'].includes(feature) ? 'rtl' : null"
                      :lang="['lemma', 'root', 'segmentation'].includes(feature) ? 'ar' : null"
                    >
                      {{ item.value }}
                    </span>
                    <span v-if="item.score !== undefined" class="candidate-score">{{ item.score }}</span>
                    <span v-if="sourceFor(token, feature, '') === item.tool" class="selected-chip">selected</span>
                  </div>
                </div>

                <EmptyCell v-else label="No returned candidate evidence" />
              </article>
            </div>
          </section>

          <section class="decision-trace">
            <button class="trace-toggle btn-ghost" type="button" @click="toggleTrace(idx)">
              {{ showTrace[idx] ? 'Hide decision trace' : 'Show decision trace' }}
            </button>
              <div v-if="showTrace[idx]" class="trace-body">
                <div v-for="row in traceRows(token)" :key="row.feature" class="trace-row">
                  <span class="trace-feature">{{ featureLabel(row.feature) }}</span>
                <ToolBadge v-if="traceSource(row)" :tool="traceSource(row)" />
                <span class="trace-value">{{ traceValue(row) }}</span>
                <small class="trace-support">
                  <span v-if="row.expert">{{ row.expert }}</span>
                  <span v-if="row.strategy">Strategy: {{ formatStrategy(row.strategy) }}</span>
                  <span v-if="row.confidence_score !== undefined">Confidence score: {{ row.confidence_score }}</span>
                  <span v-if="row.confidence_level">Confidence level: {{ row.confidence_level }}</span>
                  <span v-if="row.score_margin !== undefined && row.score_margin !== null">Decision margin: {{ row.score_margin }}</span>
                  <span v-if="row.ambiguity">Ambiguous evidence</span>
                  <span v-if="Array.isArray(row.supporting_tools) && row.supporting_tools.length">Supporting: {{ row.supporting_tools.join(' / ') }}</span>
                  <span v-if="Array.isArray(row.disagreeing_tools) && row.disagreeing_tools.length">Disagreeing: {{ row.disagreeing_tools.join(' / ') }}</span>
                  <span v-if="row.note">{{ row.note }}</span>
                </small>
              </div>
              <div v-if="!traceRows(token).length" class="null-value">Detailed decision trace was not returned for this feature.</div>
            </div>
          </section>

          <section v-if="token.conflicts.length" class="conflicts-section">
            <div class="conflicts-header">
              <span class="disagreement-badge">Disagreement</span>
              <span>{{ token.conflicts.length }} items to inspect</span>
            </div>
            <div v-for="(conflict, cidx) in token.conflicts" :key="`${idx}-conf-${cidx}`" class="conflict-item">
              <span class="conflict-feature-name">{{ featureLabel(conflict.feature) }}</span>
              <div class="conflict-values">
                <span class="conf-val"><ToolBadge :tool="conflict.tool_a" /> {{ displayValue(conflict.tool_a_value) }}</span>
                <span class="vs-sep">/</span>
                <span class="conf-val"><ToolBadge :tool="conflict.tool_b" /> {{ displayValue(conflict.tool_b_value) }}</span>
              </div>
            </div>
          </section>
        </article>
      </section>

      <section v-if="contributionGroups.length" class="section-card contribution-section">
        <h2 class="section-title">Observed contribution in this run</h2>
        <div class="contribution-grid">
          <article v-for="group in contributionGroups" :key="group.key" class="contrib-card" :class="`group-${group.key}`">
            <div class="contrib-head">
              <span class="feature-label">{{ group.label }}</span>
              <span class="group-count">{{ group.items.length }}</span>
            </div>
            <div class="contrib-items">
              <div v-for="item in group.items" :key="item.tool" class="contrib-item">
                <ToolBadge :tool="item.tool" />
                <div class="contrib-features">
                  <span v-for="feat in item.features" :key="feat" class="contrib-feat">{{ feat }}</span>
                </div>
              </div>
            </div>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import ConfidenceBadge from '@/components/badges/ConfidenceBadge.vue'
import EmptyCell from '@/components/tables/EmptyCell.vue'
import ToolBadge from '@/components/badges/ToolBadge.vue'
import ScientificChart from '@/components/charts/ScientificChart.vue'
import { analyzeAll, fusionText } from '@/api/nlpApi'
import { TOOL_COLORS, TOOL_GROUPS } from '@/constants/designTokens'
import { recordAnalysis } from '@/utils/analysisHistory'
import { formatStrategy, statusDisplay } from '@/utils/researchSemantics'

const inputText = ref('')
const loading = ref(false)
const error = ref('')
const fusionResult = ref(null)
const activeTools = ref([])
const degradedTools = ref([])
const excludedTools = ref([])
const fusionMeta = ref({})
const showTrace = ref({})
const candidateFeatures = ['lemma', 'root', 'pos', 'segmentation']
const lastDuration = ref(0)
const analyzerEvidence = ref({})
const runId = ref(0)

const visibleToolColors = computed(() =>
  Object.fromEntries(Object.entries(TOOL_COLORS).filter(([tool]) => ['camel', 'sinatools', 'stanza', 'farasa', 'alkhalil', 'udpipe', 'qalsadi'].includes(tool))),
)

const EXAMPLE_SENTENCES = [
  'قرأ الطالب الكتب في المكتبة',
  'وجدت المعلمة طالبة مجتهدة في الفصل',
  'يكتب الصحفي المقالة كل يوم',
]
const expertRoles = [
  { role: 'Segmentation anchor', tools: 'Farasa', features: 'Clitic and segment boundaries' },
  { role: 'Lexical / morphology experts', tools: 'CAMeL / SinaTools', features: 'Lemma, POS, root, and lexical morphology evidence' },
  { role: 'Syntax experts', tools: 'Stanza / UDPipe', features: 'UD POS and dependency evidence' },
  { role: 'Supporting analyzers', tools: 'Qalsadi / AlKhalil', features: 'Rule-based lexical and morphological signals' },
]


const POS_AR = {
  VERB: 'Verb',
  NOUN: 'Noun',
  ADJ: 'Adjective',
  ADP: 'Adposition',
  PRON: 'Pronoun',
  ADV: 'Adverb',
  CCONJ: 'Coordinating conjunction',
  PART: 'Particle',
  NUM: 'Number',
  DET: 'Determiner',
  PUNCT: 'Punctuation',
}
const CASE_AR = { nom: 'Nominative', acc: 'Accusative', gen: 'Genitive', Nom: 'Nominative', Acc: 'Accusative', Gen: 'Genitive' }
const TENSE_AR = { past: 'Past', perf: 'Past', pres: 'Present', imperf: 'Present', imp: 'Imperative', fut: 'Future' }
const NUMBER_AR = { singular: 'Singular', sing: 'Singular', dual: 'Dual', plural: 'Plural', plur: 'Plural' }
const DEPREL_AR = {
  nsubj: 'Subject',
  obj: 'Object',
  obl: 'Oblique',
  amod: 'Adjectival modifier',
  nmod: 'Nominal modifier',
  case: 'Case marker',
  det: 'Determiner',
  root: 'Sentence root',
  compound: 'Compound',
  advmod: 'Adverbial modifier',
  conj: 'Conjunct',
  cc: 'Coordinating conjunction',
}

const activeToolCount = computed(() => activeTools.value.length)
const activeToolsLabel = computed(() => (activeTools.value.length ? activeTools.value.join(' / ') : 'No participating tools reported'))
const totalConflicts = computed(() => fusionResult.value?.reduce((sum, token) => sum + token.conflicts.length, 0) || 0)
const averageConfidenceLabel = computed(() => {
  if (!fusionResult.value?.length) return '0%'
  const scores = fusionResult.value
    .map((token) => Number(token?.final?.confidence_score))
    .filter((value) => Number.isFinite(value))
  if (!scores.length) return '0%'
  return `${Math.round((scores.reduce((sum, value) => sum + value, 0) / scores.length) * 100)}%`
})
const confidenceTimeline = computed(() => ({
  labels: fusionResult.value?.map((_, index) => `${index + 1}`) || [],
  datasets: [
    {
      label: 'Confidence %',
      data: fusionResult.value?.map((row) => Math.round((Number(row?.final?.confidence_score) || 0) * 100)) || [],
      borderColor: '#14B8A6',
      backgroundColor: 'rgba(20, 184, 166, 0.16)',
    },
  ],
}))
const conflictSeverityChart = computed(() => {
  const counts = { high: 0, medium: 0, low: 0 }
  fusionResult.value?.forEach((token) => {
    token.conflicts?.forEach((conflict) => {
      const severity = String(conflict.severity || 'low').toLowerCase()
      if (counts[severity] !== undefined) counts[severity] += 1
    })
  })
  const labels = ['High', 'Medium', 'Low'].filter((label) => counts[label.toLowerCase()] > 0)
  return {
    labels,
    datasets: [
      {
        label: 'Conflicts',
        data: labels.map((label) => counts[label.toLowerCase()]),
        backgroundColor: ['#DC2626', '#D97706', '#14B8A6'],
      },
    ],
  }
})
const toolContributionChart = computed(() => {
  const counts = {}
  fusionResult.value?.forEach((token) => {
    Object.entries(token.sources || {}).forEach(([feature, tool]) => {
      if (!tool || tool === 'agreement') return
      counts[tool] = (counts[tool] || 0) + 1
    })
  })
  const labels = Object.keys(counts)
  return {
    labels,
    datasets: [
      {
        label: 'Fields',
        data: labels.map((label) => counts[label]),
        backgroundColor: ['#4F46E5', '#14B8A6', '#D97706', '#7C3AED', '#0EA5E9'],
      },
    ],
  }
})

const toolContributions = computed(() => {
  if (!fusionResult.value) return {}
  const contribs = {}
  fusionResult.value.forEach((token) => {
    Object.entries(token.sources || {}).forEach(([feature, tool]) => {
      if (!tool || tool === 'agreement') return
      if (!contribs[tool]) contribs[tool] = new Set()
      contribs[tool].add(featureLabel(feature))
    })
  })
  return Object.fromEntries(Object.entries(contribs).map(([tool, features]) => [tool, [...features]]))
})

const contributionGroups = computed(() => {
  const groups = [
    { key: 'morphology', label: 'Morphology / Sarf', tools: TOOL_GROUPS.morphology },
    { key: 'syntax', label: 'Syntax / Nahw', tools: TOOL_GROUPS.syntax },
    { key: 'segmentation', label: 'Segmentation / Taqti', tools: TOOL_GROUPS.segmentation },
  ]

  return groups
    .map((group) => ({
      key: group.key,
      label: group.label,
      items: group.tools
        .map((tool) => ({ tool, features: toolContributions.value[tool] || [] }))
        .filter((item) => item.features.length),
    }))
    .filter((group) => group.items.length)
})

async function runSmartAnalysis() {
  if (!inputText.value.trim()) return
  const currentRunId = ++runId.value
  loading.value = true
  error.value = ''
  fusionResult.value = null
  activeTools.value = []
  degradedTools.value = []
  excludedTools.value = []
  fusionMeta.value = {}
  showTrace.value = {}
  lastDuration.value = 0

  try {
    const started = performance.now()
    const [fusionPayload, combinedPayload] = await Promise.all([
      fusionText(inputText.value),
      analyzeAll(inputText.value).catch(() => null),
    ])
    if (currentRunId !== runId.value) return
    const data = fusionPayload
    const combinedSource = combinedPayload?.data || combinedPayload || {}
    analyzerEvidence.value = combinedSource?.tools || combinedSource?.combined?.tools || combinedSource?.combined || {}
    lastDuration.value = Math.round(performance.now() - started)
    const normalized = normalizeFusionRows(data)
    fusionResult.value = normalized.rows
    activeTools.value = normalized.activeTools
    degradedTools.value = normalized.degradedTools
    excludedTools.value = normalized.excludedTools
    fusionMeta.value = normalized.meta
    recordAnalysis({
      page: 'Fusion',
      text: inputText.value.trim(),
      summary: `${normalized.rows.length} tokens | ${totalConflicts.value} conflicts | ${averageConfidenceLabel.value} avg confidence`,
    })
  } catch (e) {
    if (currentRunId === runId.value) error.value = e?.response?.data?.detail || e?.message || 'Unable to reach the backend. Make sure FastAPI is running on localhost:8000.'
  } finally {
    if (currentRunId === runId.value) loading.value = false
  }
}

function runExample(example) {
  inputText.value = example
  runSmartAnalysis()
}

function normalizeFusionRows(payload) {
  const source = payload?.data || payload || {}
  const tools = source?.tools || {}
  const rows = Array.isArray(source)
    ? source
    : Array.isArray(source?.fusion)
      ? source.fusion
      : Array.isArray(source?.fusion_result?.fusion)
        ? source.fusion_result.fusion
        : Array.isArray(source?.fusion_result)
          ? source.fusion_result
          : []

  return {
    rows: rows.map((row, index) => {
      const legacyFinal = row?.fusion || {}
      const final = {
        ...(row?.final || {}),
        confidence_score: row?.final?.confidence_score ?? row?.confidence_score ?? legacyFinal.confidence ?? 0,
        confidence_level: row?.final?.confidence_level || row?.confidence_level || legacyFinal.confidence_level || 'low',
      }
      if (!final.pos && legacyFinal.final_pos) final.pos = legacyFinal.final_pos
      if (!final.lemma && legacyFinal.final_lemma) final.lemma = legacyFinal.final_lemma
      if (!final.segmentation && legacyFinal.final_segmentation) final.segmentation = legacyFinal.final_segmentation

      return {
        index,
        word: row?.word || row?.surface || row?.token || `#${index + 1}`,
        final,
        sources: row?.sources || legacyFinal.chosen_sources || {},
        expert_decisions: row?.expert_decisions || {},
        expert_summary: row?.expert_summary || {},
        fusion_mode: row?.fusion_mode || '',
        conflicts: Array.isArray(row?.conflicts) ? row.conflicts : [],
        notes: Array.isArray(row?.notes) ? row.notes : [],
        decision_trace: Array.isArray(row?.decision_trace) ? row.decision_trace : [],
        evidence: row?.evidence || {},
      }
    }),
    activeTools: source?.active_tools || source?.meta?.active_tools || [],
    degradedTools: source?.meta?.degraded_tools || Object.entries(tools).filter(([, item]) => {
      const display = statusDisplay(item?.status)
      return ['degraded', 'error', 'unavailable'].includes(display.group)
    }).map(([tool]) => tool),
    excludedTools: Object.entries(tools).filter(([, item]) => statusDisplay(item?.status).group === 'excluded').map(([tool]) => tool),
    meta: source?.meta || {},
  }
}

function toggleTrace(index) {
  showTrace.value = { ...showTrace.value, [index]: !showTrace.value[index] }
}

function traceRows(token) {
  if (token.decision_trace.length) return token.decision_trace
  return []
}

const FUSION_ELIGIBILITY = {
  lemma: ['camel', 'sinatools', 'alkhalil', 'stanza', 'udpipe', 'qalsadi'],
  root: ['camel', 'sinatools', 'alkhalil'],
  pos: ['camel', 'sinatools', 'alkhalil', 'stanza', 'udpipe'],
  segmentation: ['farasa'],
}

function candidateEvidence(tokenIndex, feature) {
  const token = fusionResult.value?.[tokenIndex]
  const backendCandidates = token?.expert_decisions?.[feature]?.candidates
  if (Array.isArray(backendCandidates) && backendCandidates.length) {
    return backendCandidates.flatMap((candidate, index) => {
      const tools = Array.isArray(candidate.tools) ? candidate.tools : []
      const label = tools.join(' / ') || `candidate ${index + 1}`
      return [{
        tool: tools[0] || '',
        label,
        value: candidate.value,
        score: candidate.score,
      }]
    })
  }

  return (FUSION_ELIGIBILITY[feature] || []).flatMap((tool) => {
    const payload = analyzerEvidence.value?.[tool]
    const token = Array.isArray(payload?.tokens) ? payload.tokens[tokenIndex] : null
    if (!token) return []

    const analysis = Array.isArray(token.analyses) && token.analyses.length ? token.analyses[0] : token
    let value = ''

    if (feature === 'pos') {
      value = analysis?.pos || analysis?.upos || token?.pos || token?.upos || ''
    } else if (feature === 'segmentation') {
      const segmentation = analysis?.segmentation || token?.segmentation || token?.segments || token?.parts
      value = Array.isArray(segmentation) ? segmentation.join(' + ') : segmentation
    } else {
      value = analysis?.[feature] || token?.[feature] || ''
    }

    return hasValue(value) ? [{ tool, value: String(value) }] : []
  })
}

function sourceFor(token, feature, fallback) {
  return token.expert_decisions?.[feature]?.primary_source || token.sources?.[feature] || fallback
}

function hasValue(value) {
  return value !== null && value !== undefined && value !== '' && value !== 'null'
}

function segmentationList(token) {
  const value = token.final.segmentation
  if (Array.isArray(value)) return value.filter(hasValue)
  if (typeof value === 'string' && value.trim()) return value.split(/[+ ]/).filter(Boolean)
  return []
}

function dependencyLabel(dep) {
  if (!dep) return ''
  const deprel = typeof dep === 'string' ? dep : dep.deprel
  if (!deprel || deprel === 'root') return deprel === 'root' ? DEPREL_AR.root : ''
  return DEPREL_AR[deprel] || deprel
}

function displayValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(' + ') : 'Not returned'
  if (value && typeof value === 'object') return JSON.stringify(value)
  return hasValue(value) ? String(value) : 'Not returned'
}

function decisionFor(token, feature) {
  return token?.expert_decisions?.[feature] || {}
}

function primarySourceLabel(token, feature) {
  const source = sourceFor(token, feature, '')
  return source ? `Primary source: ${source}` : 'Primary source not returned'
}

function traceValue(row) {
  return displayValue(row.chosen_value ?? row.value)
}

function traceSource(row) {
  return row.primary_source || row.source || row.tool || ''
}

function featureLabel(feature) {
  const labels = {
    lemma: 'Lemma',
    root: 'Root',
    root_type: 'Root type',
    pos: 'POS',
    gloss: 'Gloss',
    gender: 'Gender',
    number: 'Number',
    tense: 'Tense',
    case: 'Case',
    definite: 'Definiteness',
    voice: 'Voice',
    dependency: 'Dependency',
    segmentation: 'Segmentation',
  }
  return labels[feature] || feature
}

function rootTypeLabel(value) {
  if (value === 'triliteral') return 'Triliteral'
  if (value === 'biliteral') return 'Biliteral'
  if (value === 'monoliteral') return 'Monoliteral'
  return value
}

function genderLabel(value) {
  return value === 'masculine' || value === 'masc' || value === 'Masc' ? 'Masculine' : 'Feminine'
}

function definiteLabel(value) {
  return value === 'yes' || value === 'Def' ? 'Definite' : 'Indefinite'
}

function voiceLabel(value) {
  return value === 'passive' || value === 'Pass' ? 'Passive' : 'Active'
}
</script>

<style scoped>
.smart-page {
  display: grid;
  gap: 18px;
}

.smart-hero {
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: end;
}

.tool-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  max-width: 420px;
}

.input-section {
  margin-bottom: 0;
}

.input-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  margin-top: 8px;
}

.arabic-input {
  flex: 1;
  min-height: 74px;
  padding: 12px 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-control);
  color: var(--c-text-primary);
  background: var(--c-surface);
  font-size: 20px;
  line-height: 1.7;
  resize: vertical;
  outline: none;
}

.arabic-input:focus {
  border-color: var(--c-accent);
  box-shadow: 0 0 0 3px var(--c-accent-light);
}

.analyze-btn {
  min-width: 138px;
  min-height: 52px;
  padding: 12px 24px;
  border-radius: var(--radius-control);
  color: white;
  background: linear-gradient(135deg, var(--c-text-primary), var(--c-accent));
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
}

.analyze-btn:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.spinner,
.loading-spinner-large {
  display: inline-block;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
}

.examples-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.examples-label {
  color: var(--c-text-muted);
  font-size: 12px;
}

.example-chip {
  padding: 5px 12px;
  border: 1px solid var(--c-border);
  border-radius: 999px;
  color: var(--c-text-secondary);
  background: var(--c-page-bg);
  cursor: pointer;
  font-size: 13px;
}

.error-banner {
  padding: 12px 16px;
  border: 1px solid var(--c-conf-low-border);
  border-radius: var(--radius-control);
  color: var(--c-conf-low-text);
  background: var(--c-conf-low-bg);
}

.smart-loading {
  min-height: 180px;
}

.loading-spinner-large {
  width: 40px;
  height: 40px;
  margin: 0 auto 1rem;
  border: 3px solid var(--c-border);
  border-top-color: var(--c-accent);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.summary-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  padding: 1rem 1.25rem;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-control);
  background: var(--c-page-bg);
}

.fusion-participation-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.participation-card {
  display: grid;
  gap: 6px;
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.participation-card strong {
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--c-text-primary);
  font-size: 13px;
  font-weight: 650;
}

.summary-item {
  display: grid;
  gap: 2px;
  min-width: 150px;
}

.summary-label {
  color: var(--c-text-muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.summary-item strong {
  color: var(--c-text-primary);
  font-size: 16px;
  font-weight: 600;
}

.analysis-visual-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.tool-list {
  direction: ltr;
  text-align: right;
}

.conflict-count {
  color: var(--c-segment-text) !important;
}

.tokens-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 16px;
}

.token-card {
  padding: 1.25rem;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-control);
  background: var(--c-surface);
}

.token-card.has-conflicts {
  border-color: var(--c-conf-med-border);
  background: linear-gradient(180deg, var(--c-surface) 0%, var(--c-conf-med-bg) 140%);
}

.token-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 12px;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--c-page-bg);
}

.token-word {
  color: var(--c-text-primary);
  font-size: 24px;
  font-weight: 700;
}

.token-meta {
  margin-top: 6px;
  color: var(--c-text-muted);
  font-size: 12px;
}

.features-grid {
  display: grid;
  gap: 10px;
}

.feature-item {
  display: grid;
  gap: 5px;
}

.feature-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.feature-value {
  color: var(--c-text-primary);
  font-size: 16px;
  font-weight: 500;
}

.source-line {
  color: var(--c-text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.root-value {
  color: var(--c-morphology-text);
  letter-spacing: 0.08em;
}

.root-type-badge {
  width: fit-content;
  padding: 2px 8px;
  border-radius: 999px;
  color: var(--c-morphology-text);
  background: var(--c-morphology-bg);
  font-size: 11px;
  font-weight: 500;
}

.pos-badge,
.morph-pill,
.seg-pill,
.dep-rel {
  width: fit-content;
  border-radius: var(--radius-chip);
  font-size: 12px;
  font-weight: 500;
}

.pos-badge {
  padding: 4px 12px;
  border: 1px solid var(--c-accent-border);
  color: var(--c-accent-text);
  background: var(--c-accent-light);
}

.gloss-value {
  direction: ltr;
  color: var(--c-text-secondary);
  font-size: 14px;
  font-style: italic;
}

.morph-pills,
.seg-pills,
.dep-row,
.conflict-values {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}

.morph-pill {
  padding: 4px 10px;
  color: var(--c-text-secondary);
  background: var(--c-page-bg);
}

.stanza-pill {
  color: var(--c-conf-high-text);
  background: var(--c-conf-high-bg);
}

.seg-pill {
  padding: 4px 10px;
  border: 1px solid var(--c-morphology-border);
  color: var(--c-morphology-text);
  background: var(--c-morphology-bg);
}

.dep-rel {
  padding: 4px 10px;
  color: var(--c-conf-high-text);
  background: var(--c-conf-high-bg);
}

.dep-head {
  color: var(--c-text-primary);
  font-size: 14px;
  font-weight: 500;
}

.decision-trace {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--c-page-bg);
}

.trace-toggle {
  padding: 0;
  color: var(--c-text-secondary);
  background: transparent;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
}

.trace-body {
  display: grid;
  gap: 6px;
  margin-top: 10px;
}

.trace-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 5px 0;
  border-bottom: 1px dashed var(--c-page-bg);
  font-size: 12px;
}

.trace-feature {
  min-width: 86px;
  color: var(--c-text-secondary);
  font-weight: 600;
}

.trace-value {
  color: var(--c-text-primary);
  font-weight: 600;
}

.trace-support {
  color: var(--c-text-muted);
}

.conflicts-section {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--c-conf-med-border);
  border-radius: var(--radius-control);
  background: var(--c-conf-med-bg);
}

.conflicts-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  color: var(--c-conf-med-text);
  font-size: 12px;
  font-weight: 600;
}

.conflict-item {
  display: grid;
  gap: 4px;
  margin-top: 8px;
}

.conflict-feature-name {
  color: var(--c-conf-med-text);
  font-size: 11px;
  font-weight: 600;
}

.conf-val {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--c-text-secondary);
  font-size: 12px;
  font-weight: 500;
}

.vs-sep {
  color: var(--c-segment-text);
  font-weight: 700;
}

.contribution-section {
  margin-bottom: 0;
}

.section-title {
  margin: 0 0 1rem;
  color: var(--c-text-primary);
  font-size: 16px;
  font-weight: 700;
}

.contribution-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}

.contrib-card {
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-control);
  background: var(--c-page-bg);
}

.contrib-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.group-count {
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--c-surface);
  color: var(--c-text-secondary);
  font-size: 12px;
  font-weight: 600;
}

.contrib-items {
  display: grid;
  gap: 8px;
}

.contrib-item {
  display: grid;
  gap: 6px;
}

.contrib-features {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.contrib-feat {
  padding: 3px 8px;
  border: 1px solid var(--c-border);
  border-radius: 999px;
  color: var(--c-text-secondary);
  background: var(--c-surface);
  font-size: 11px;
  font-weight: 500;
}

.research-method-panel {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1.45fr);
  gap: 18px;
  padding: 22px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}
.method-copy h2 { margin: 6px 0 10px; font-size: clamp(1.15rem, 2vw, 1.55rem); }
.method-copy p { margin: 0; color: var(--c-text-secondary); line-height: 1.7; }
.expert-map { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.expert-role { display: grid; gap: 5px; padding: 14px; border: 1px solid var(--c-border); border-radius: var(--radius-control); background: var(--c-bg-subtle, #f8fafc); }
.expert-role-label { color: var(--c-text-secondary); font-size: .76rem; text-transform: uppercase; letter-spacing: .05em; }
.expert-role small { color: var(--c-text-secondary); line-height: 1.45; }

@media (max-width: 860px) {
  .research-method-panel { grid-template-columns: 1fr; }
  .smart-hero,
  .input-row {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .tool-legend {
    justify-content: flex-start;
  }

  .analyze-btn {
    width: 100%;
  }

  .analysis-visual-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .expert-map { grid-template-columns: 1fr; }
  .research-method-panel { padding: 16px; }
  .tokens-grid {
    grid-template-columns: 1fr;
  }
}

.fusion-method-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}.decision-policy h2{margin:6px 0 8px}.decision-policy p{color:var(--c-text-secondary);line-height:1.7}.role-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px}.role-card{display:grid;gap:3px;padding:12px;border:1px solid var(--c-border);border-radius:8px;background:var(--c-page-bg)}.role-card span{color:var(--c-accent-text);font-size:13px}.role-card small{color:var(--c-text-secondary);line-height:1.5}@media(max-width:900px){.fusion-method-grid{grid-template-columns:1fr}.role-grid{grid-template-columns:1fr}}

.candidate-evidence{margin-top:14px;padding:13px;border:1px solid var(--c-border);border-radius:10px;background:var(--c-page-bg)}
.candidate-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:10px}
.candidate-head small{max-width:460px;color:var(--c-text-secondary);text-align:right;line-height:1.45}
.candidate-feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
.candidate-feature{min-width:0;padding:10px;border:1px solid var(--c-border);border-radius:8px;background:var(--c-surface)}
.candidate-feature>strong{display:block;margin-bottom:8px;font-size:12px}
.candidate-list{display:grid;gap:6px}.candidate-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:7px;align-items:center;min-width:0}
.candidate-row>span:nth-child(2){overflow-wrap:anywhere}.selected-mark{color:var(--c-agreement-text);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
@media(max-width:760px){.candidate-head{flex-direction:column}.candidate-head small{text-align:left}.candidate-feature-grid{grid-template-columns:1fr}}

/* FINAL FUSION RESEARCH PASS */
.smart-page {
  width: min(96vw, 1500px);
}

.smart-hero {
  grid-template-columns: minmax(0, 1fr);
  align-items: start;
}

.tool-legend {
  margin-top: 18px;
  justify-content: flex-start;
  max-width: 100%;
}

.analyze-btn {
  background: var(--c-accent);
  box-shadow: none;
}

.summary-bar {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  padding: 0;
  border: 0;
  background: transparent;
}

.summary-item {
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.conflict-count {
  color: #7a5a2e !important;
}

.fusion-method-grid {
  grid-template-columns: minmax(0, 1.2fr) minmax(340px, .8fr);
  gap: 16px;
}

.role-card {
  border-left: 3px solid var(--c-accent);
  background: #f8fafc;
}

.tokens-grid {
  grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
  gap: 18px;
}

.token-card {
  border-radius: var(--radius-card);
  border-color: #d9e2ec;
  box-shadow: 0 8px 22px rgba(15, 23, 42, .04);
}

.token-card.has-conflicts {
  border-color: #c9a46a;
  background: linear-gradient(180deg, #ffffff 0%, #fffaf2 135%);
}

.token-header {
  border-bottom-color: var(--c-border);
}

.token-word {
  font-size: 26px;
}

.features-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.feature-item {
  min-width: 0;
  padding: 10px;
  border: 1px solid #eef2f6;
  border-radius: 10px;
  background: #fbfdff;
}

.feature-item.full-width {
  grid-column: 1 / -1;
}

.feature-label,
.feature-label-row .feature-label {
  color: #64748b;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.feature-value,
.gloss-value,
.dep-head {
  overflow-wrap: anywhere;
  line-height: 1.5;
}

.pos-badge {
  border-color: #c9d8e6;
  background: #eef4f9;
  color: #244a73;
}

.root-type-badge,
.morph-pill,
.seg-pill,
.dep-rel {
  border: 1px solid #d9e2ec;
  background: #f8fafc;
  color: #475569;
}

.stanza-pill,
.dep-rel {
  color: #365f56;
  background: #f0f7f4;
  border-color: #c9dfd8;
}

/* Candidate evidence */
.candidate-evidence-panel {
  margin-top: 14px;
  padding: 14px;
  border: 1px solid var(--c-border);
  border-radius: 12px;
  background: #f8fafc;
}

.candidate-panel-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: flex-start;
  margin-bottom: 12px;
}

.candidate-panel-head p {
  margin: 3px 0 0;
  color: var(--c-text-secondary);
  font-size: 12px;
  line-height: 1.55;
}

.candidate-note {
  flex: 0 0 auto;
  padding: 5px 9px;
  border: 1px solid #d9e2ec;
  border-radius: 999px;
  background: #fff;
  color: #64748b;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .035em;
  text-transform: uppercase;
}

.candidate-feature-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.candidate-feature-card {
  min-width: 0;
  padding: 10px;
  border: 1px solid #d9e2ec;
  border-radius: 10px;
  background: #fff;
}

.candidate-feature-card header {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}

.candidate-feature-card header strong {
  color: var(--c-text-primary);
  font-size: 12px;
}

.candidate-list {
  display: grid;
  gap: 6px;
}

.candidate-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 7px;
  align-items: center;
  min-width: 0;
  padding: 7px;
  border: 1px solid #edf2f7;
  border-radius: 8px;
  background: #f8fafc;
}

.candidate-row.selected {
  border-color: #b8d3c8;
  background: #f0f7f4;
}

.candidate-value {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 12.5px;
  font-weight: 650;
  color: var(--c-text-primary);
}

.candidate-score {
  padding: 2px 6px;
  border-radius: 999px;
  background: #fff;
  border: 1px solid #d9e2ec;
  color: #64748b;
  font-size: 10px;
  font-weight: 800;
}

.candidate-value.arabic {
  text-align: right;
  font-size: 14px;
}

.selected-chip {
  padding: 3px 7px;
  border-radius: 999px;
  background: #ffffff;
  color: #365f56;
  border: 1px solid #b8d3c8;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}

/* Decision trace and disagreements */
.decision-trace {
  border-top-color: var(--c-border);
}

.trace-row {
  display: grid;
  grid-template-columns: 92px auto minmax(0, 1fr);
  gap: 8px;
  align-items: center;
}

.trace-support {
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  padding-left: 0;
  line-height: 1.5;
}

.conflicts-section {
  border-color: #d9c28f;
  background: #fffaf2;
}

.conflicts-header {
  color: #7a5a2e;
}

.disagreement-badge {
  padding: 3px 8px;
  border-radius: 999px;
  background: #f1e5ca;
  color: #7a5a2e;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .04em;
}

.conflict-item {
  padding: 8px;
  border: 1px solid #ead9b7;
  border-radius: 8px;
  background: #fff;
}

.conflict-feature-name {
  color: #7a5a2e;
}

.conf-val {
  min-width: 0;
  overflow-wrap: anywhere;
}

.vs-sep {
  color: #7a5a2e;
}

.contribution-section {
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
}

.contrib-card {
  background: #f8fafc;
}

@media (max-width: 1050px) {
  .smart-page {
    width: min(100% - 24px, 1240px);
  }

  .summary-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .fusion-method-grid,
  .fusion-participation-grid,
  .features-grid,
  .candidate-feature-grid {
    grid-template-columns: 1fr;
  }

  .tokens-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .summary-bar {
    grid-template-columns: 1fr;
  }

  .candidate-panel-head {
    flex-direction: column;
  }

  .trace-row {
    grid-template-columns: 1fr;
  }
}

</style>
