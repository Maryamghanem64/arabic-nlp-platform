<template>
  <div class="page-wrap analyze-page page-stack">
    <section class="hero-band analyze-hero">

      <div class="hero-content">
        <span class="eyebrow">Analyzer evidence laboratory</span>
        <h1 class="hero-title">Inspect structured Arabic NLP evidence at token level.</h1>
        <p class="hero-copy">
          Run one analyzer for a capability-focused inspection or execute the combined path. Results preserve analyzer evidence and expose missing fields without inventing values.
        </p>
      </div>
    </section>

    <section class="panel panel-pad">
      <div class="section-head">
        <div>
          <h2 class="section-title">Analysis Input</h2>
          <p class="section-subtitle">{{ tokenEstimate }} token{{ tokenEstimate === 1 ? '' : 's' }} estimated</p>
        </div>
        <div class="actions-row compact-actions">
          <button class="btn btn-subtle" @click="loadSample">Sample</button>
          <button class="btn btn-subtle" @click="clear">Clear</button>
        </div>
      </div>

      <textarea
        v-model="inputText"
        class="textarea arabic"
        dir="rtl"
        lang="ar"
        placeholder="Enter Arabic text here..."
      ></textarea>

      <div class="run-row">
        <button class="btn btn-primary" :disabled="loading || !inputText.trim()" @click="analyze">
          {{ loading ? 'Analyzing...' : 'Run Analysis' }}
        </button>
        <button class="btn btn-secondary" :disabled="!hasResults" @click="copyCurrentJson">Copy JSON</button>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="jsonExportHref" @click="guardExport">Export JSON</a>
        <a class="btn btn-secondary" :class="{ disabled: !hasResults }" :href="csvExportHref" @click="guardExport">Export CSV</a>
        <span v-if="copied" class="copy-note">Copied</span>
      </div>
    </section>

    <section class="panel panel-pad selector-panel">
      <div class="section-head">

        <div>
          <h2 class="section-title">Tool Selector</h2>
          <p class="section-subtitle">The list is driven by the shared tool config and live backend status.</p>
        </div>
      </div>

      <div class="selector-grid">
        <button
          class="selector-card all-tools"
          :class="{ active: selectedTool === 'all' }"
          type="button"
          @click="selectTool('all')"
        >
          <span class="selector-dot selector-dot--neutral"></span>
          <div class="selector-copy">
            <span class="selector-label">All tools</span>
            <span class="selector-subtitle">Combined analysis</span>
          </div>
        </button>

        <button
          v-for="tool in toolOptions"
          :key="tool.key"
          class="selector-card"
          :class="{ active: selectedTool === tool.key }"
          type="button"
          @click="selectTool(tool.key)"
        >
          <span class="selector-dot" :style="{ backgroundColor: tool.color }"></span>
          <div class="selector-copy">
            <div class="selector-row">
              <span class="selector-label">{{ tool.label }}</span>
              <span v-if="statusBadge(tool.key).show" :class="['pill', statusBadge(tool.key).className]">
                {{ statusBadge(tool.key).label }}
              </span>
            </div>
            <span class="selector-subtitle">{{ toolRole(tool.key) }}</span>
          </div>
        </button>
      </div>

      <div v-if="selectionNotice" class="selection-notice" :class="selectionNotice.kind">
        <strong>{{ selectionNotice.title }}</strong>
        <p>{{ selectionNotice.message }}</p>
        <div v-if="selectionNotice.pending" class="actions-row compact-actions">
          <button class="btn btn-secondary" @click="confirmPendingSelection">Use tool anyway</button>
          <button class="btn btn-subtle" @click="cancelPendingSelection">Cancel</button>
        </div>
      </div>
    </section>

    <section class="selected-capability-panel">
      <div>
        <span class="eyebrow">Selected analysis scope</span>
        <h2>{{ selectedToolMeta.label }}</h2>
        <p>{{ selectedCapability.description }}</p>
      </div>
      <div class="capability-tags">
        <span v-for="feature in selectedCapability.features" :key="feature">{{ feature }}</span>
      </div>
    </section>

    <section v-if="hasResults && !loading" class="analysis-summary-grid" aria-label="Analysis summary">
      <article class="analysis-summary-card"><span>Tokens analyzed</span><strong>{{ currentRows.length }}</strong></article>
      <article class="analysis-summary-card"><span>Analysis scope</span><strong>{{ selectedTool === 'all' ? 'Combined' : selectedToolMeta.label }}</strong></article>
      <article class="analysis-summary-card"><span>Average confidence</span><strong>{{ averageConfidenceLabel }}</strong></article>
      <article class="analysis-summary-card"><span>Absent expected evidence</span><strong>{{ missingEvidenceCount }}</strong></article>
    </section>

    <section v-if="hasResults && !loading" class="analysis-visual-grid">
      <ScientificChart
        type="line"
        title="Token Confidence Timeline"
        subtitle="Confidence by token position for the current analysis."
        badge="Timeline"
        :labels="tokenTimelineChart.labels"
        :datasets="tokenTimelineChart.datasets"
        :height="260"
        aria-label="Token confidence timeline"
        empty-title="No token timeline"
        empty-text="The current result set does not include confidence scores."
      />

      <ScientificChart
        type="bar"
        title="POS Distribution"
        subtitle="Distribution of part-of-speech labels in the current result."
        badge="POS"
        :labels="posDistributionChart.labels"
        :datasets="posDistributionChart.datasets"
        :height="260"
        aria-label="POS distribution chart"
        empty-title="No POS distribution"
        empty-text="The current result set does not include POS labels."
      />
    </section>

    <section v-if="hasResults && !loading" class="analysis-visual-grid analysis-visual-grid--wide">
      <HeatmapMatrix
        title="Expected Evidence Matrix"
        subtitle="Presence of fields expected from the selected analyzer capability."
        badge="Matrix"
        :rows="heatmapRows"
        :cols="heatmapCols"
        :values="heatmapValues"
        empty-title="No evidence matrix"
        empty-text="The current run does not expose enough structured evidence to build a matrix."
      />
    </section>

    <div v-if="statusError && !toolStatusesLoaded" class="error-state">
      <div>
        <strong>Backend status unavailable</strong>
        <p>{{ statusError.message || 'Could not load tool availability from GET /.' }}</p>
      </div>
    </div>

    <div v-if="loading" class="loading-state analysis-loading">
      <div class="loading-stack">
        <span class="spinner--dark" aria-hidden="true"></span>
        <span>Running analyzer tools...</span>
      </div>
    </div>

    <div v-if="error" class="error-state">
      <div>
        <strong>Analysis failed</strong>
        <p>{{ error }}</p>
        <button class="btn btn-secondary" @click="analyze">Retry</button>
      </div>
    </div>

    <template v-if="hasResults && !loading">
      <section class="panel panel-pad">
        <div class="analysis-tabs">
          <button v-for="tab in tabs" :key="tab.key" type="button" :class="{ active: activeTab === tab.key }" @click="activeTab = tab.key">
            {{ tab.label }}
          </button>
        </div>

        <div v-if="activeTab === 'results'" class="tab-panel">
          <div v-if="selectedTool === 'all'" class="capability-results">
            <section v-for="section in groupedToolSections" :key="section.key" class="capability-group">
              <div class="capability-group-head">
                <div>
                  <span class="feature-label">{{ section.kicker }}</span>
                  <h3>{{ section.title }}</h3>
                  <p>{{ section.description }}</p>
                </div>
                <span class="group-count">{{ section.tools.length }} analyzer{{ section.tools.length === 1 ? '' : 's' }}</span>
              </div>

              <div :class="['capability-tool-grid', section.layout]">
                <article v-for="tool in section.tools" :key="tool.key" class="section-card tool-result-card">
              <div class="tool-result-head" :style="{ borderTopColor: tool.color }">
                <div class="tool-result-title">
                  <span class="tool-result-bar" :style="{ backgroundColor: tool.color }"></span>
                  <div>
                    <h3>{{ tool.label }}</h3>
                    <p>{{ toolRole(tool.key) }}</p>
                  </div>
                </div>
                <span :class="['pill', statusPill(resultStatus(tool.key))]">{{ statusLabel(resultStatus(tool.key)) }}</span>
              </div>

              <div
                v-if="isRenderableToolStatus(resultStatus(tool.key)) && toolRows(tool.key).length"
                class="compact-evidence-table capability-specific-table"
              >
                <table>
                  <thead>
                    <tr>
                      <th>Token</th>
                      <th
                        v-for="column in toolTableColumns(tool.key)"
                        :key="`${tool.key}-header-${column.key}`"
                      >
                        {{ column.label }}
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    <tr
                      v-for="row in toolRows(tool.key)"
                      :key="`${tool.key}-${row.index}`"
                    >
                      <td class="arabic token-cell" dir="rtl" lang="ar">
                        {{ row.surface }}
                      </td>

                      <td
                        v-for="column in toolTableColumns(tool.key)"
                        :key="`${tool.key}-${row.index}-${column.key}`"
                        :data-label="column.label"
                      >
                        <span
                          :class="toolCellClass(tool.key, column.key, row)"
                          :dir="toolCellDirection(column.key)"
                          :lang="toolCellDirection(column.key) === 'rtl' ? 'ar' : null"
                        >
                          {{ toolCellValue(tool.key, column.key, row) }}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-else-if="isRenderableToolStatus(resultStatus(tool.key))" class="empty-state">{{ emptyToolOutputLabel(tool.key) }}</div>
              <div v-else class="status-card">
                <strong>{{ statusLabel(resultStatus(tool.key)) }}</strong>
                <p>{{ resultReason(tool.key) || statusDescription(resultStatus(tool.key)) }}</p>
              </div>
            </article>

              </div>
            </section>
          </div>

          <article v-else class="section-card single-result-card">
            <div class="tool-result-head" :style="{ borderTopColor: selectedToolMeta.color }">
              <div class="tool-result-title">
                <span class="tool-result-bar" :style="{ backgroundColor: selectedToolMeta.color }"></span>
                <div>
                  <h3>{{ selectedToolMeta.label }}</h3>
              <p>{{ toolRole(selectedTool) }}</p>
                </div>
              </div>
              <span :class="['pill', statusPill(resultStatus(selectedTool))]">{{ statusLabel(resultStatus(selectedTool)) }}</span>
            </div>

            <div
              v-if="isRenderableToolStatus(resultStatus(selectedTool)) && toolRows(selectedTool).length"
              class="compact-evidence-table capability-specific-table"
            >
              <table>
                <thead>
                  <tr>
                    <th>Token</th>
                    <th
                      v-for="column in toolTableColumns(selectedTool)"
                      :key="`${selectedTool}-header-${column.key}`"
                    >
                      {{ column.label }}
                    </th>
                  </tr>
                </thead>

                <tbody>
                  <tr
                    v-for="row in toolRows(selectedTool)"
                    :key="`${selectedTool}-${row.index}`"
                  >
                    <td class="arabic token-cell" dir="rtl" lang="ar">
                      {{ row.surface }}
                    </td>

                    <td
                      v-for="column in toolTableColumns(selectedTool)"
                      :key="`${selectedTool}-${row.index}-${column.key}`"
                      :data-label="column.label"
                    >
                      <span
                        :class="toolCellClass(selectedTool, column.key, row)"
                        :dir="toolCellDirection(column.key)"
                        :lang="toolCellDirection(column.key) === 'rtl' ? 'ar' : null"
                      >
                        {{ toolCellValue(selectedTool, column.key, row) }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="status-card">
              <strong>{{ statusLabel(resultStatus(selectedTool)) }}</strong>
              <p>{{ resultReason(selectedTool) || statusDescription(resultStatus(selectedTool)) }}</p>
            </div>
          </article>
        </div>

        <div v-if="activeTab === 'fusion' && selectedTool === 'all'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Fusion Output</h2>
              <p class="section-subtitle">Fusion is shown only in All tools mode, as requested.</p>
            </div>
            <button class="btn btn-secondary" @click="loadFusion">Refresh Fusion</button>
          </div>

          <div v-if="fusionLoading" class="info-state">
            Analyzer outputs are ready. Fusion evidence is still being prepared.
          </div>
          <div v-if="fusionError" class="info-state info-state--warn">
            Fusion evidence is unavailable for this run: {{ fusionError }}
          </div>

          <div v-if="fusionRows.length" class="table-scroll fusion-table">
            <table>
              <thead>
                <tr>
                  <th>Word</th>
                  <th>Lemma</th>
                  <th>Root</th>
                  <th>POS</th>
                  <th>Segmentation</th>
                  <th>Confidence</th>
                  <th>Sources</th>
                  <th>Conflicts</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in fusionRows" :key="`${row.word}-${row.index}`">
                  <td class="arabic" dir="rtl" lang="ar">{{ row.word }}</td>
                  <td>
                    <span v-if="hasTokenValue(row.final.lemma)" class="arabic-value">{{ displayTokenValue('lemma', row.final.lemma) }}</span>
                    <EmptyCell v-else />
                  </td>
                  <td>
                    <span v-if="hasTokenValue(row.final.root)" class="arabic-value">{{ displayTokenValue('root', row.final.root) }}</span>
                    <EmptyCell v-else />
                  </td>
                  <td>
                    <span v-if="hasTokenValue(row.final.pos)" class="ltr-value">{{ displayTokenValue('pos', row.final.pos) }}</span>
                    <EmptyCell v-else />
                  </td>
                  <td>
                    <span v-if="hasTokenValue(row.final.segmentation)" class="arabic-value">{{ displayTokenValue('segmentation', row.final.segmentation) }}</span>
                    <EmptyCell v-else />
                  </td>
                  <td><span :class="['pill', confidencePill(row.final.confidence_level)]">{{ row.final.confidence_level || 'low' }}</span></td>
                  <td>
                    <div class="chip-row">
                      <span
                        v-for="chip in fusionSourceChips(row.sources)"
                        :key="`${row.index}-${chip.label}`"
                        class="source-chip"
                        :style="{ backgroundColor: chip.color }"
                      >
                        {{ chip.label }}
                      </span>
                    </div>
                  </td>
                  <td>
                    <div v-if="row.conflicts?.length" class="conflict-stack">
                      <article v-for="(conflict, cIndex) in row.conflicts" :key="`${row.index}-conflict-${cIndex}`" class="conflict-mini-card">
                        <div class="conflict-mini-head">
                          <span class="conflict-mini-field">{{ fieldLabel(conflict.feature) }}</span>
                          <span :class="['pill', severityPill(conflict.severity)]">{{ conflict.severity || 'low' }}</span>
                        </div>
                        <div class="conflict-mini-line">
                          <strong>Tools</strong>
                          <span>{{ conflict.tool_a }} vs {{ conflict.tool_b }}</span>
                        </div>
                        <div class="conflict-mini-line">
                          <strong>Values</strong>
                          <span>{{ displayConflictValue(conflict.tool_a_value) }} / {{ displayConflictValue(conflict.tool_b_value) }}</span>
                        </div>
                      </article>
                    </div>
                    <span v-else class="no-conflict-label">No conflicts</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else-if="!fusionLoading" class="empty-state">Fusion output is not available for this run.</div>
        </div>

        <div v-if="activeTab === 'json'" class="tab-panel">
          <div class="section-head">
            <div>
              <h2 class="section-title">Formatted JSON</h2>
              <p class="section-subtitle">Raw response for reproducibility and debugging.</p>
            </div>
          </div>
          <pre class="json-panel"><code>{{ prettyJson }}</code></pre>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import EmptyCell from '../components/tables/EmptyCell.vue'
import ScientificChart from '../components/charts/ScientificChart.vue'
import HeatmapMatrix from '../components/charts/HeatmapMatrix.vue'
import { analyzeAll, analyzeTool, exportUrl, fusionText } from '../api/nlpApi'
import { TOOL_CONFIG, TOOL_KEYS, toolOrder } from '../config/tools'
import { useToolStatus } from '../composables/useToolStatus'
import { canonicalToken } from '../utils/tokenModel'
import { recordAnalysis } from '../utils/analysisHistory'
import { canRenderToolEvidence, missingValueLabel, statusDisplay, toolRole } from '../utils/researchSemantics'

const route = useRoute()
const inputText = ref('')

const loading = ref(false)
const fusionLoading = ref(false)
const error = ref('')
const fusionError = ref('')
const rawResults = ref(null)
const fusionPayload = ref(null)
const selectedTool = ref('all')
const activeTab = ref('results')
const copied = ref(false)
const pendingTool = ref('')
const selectionNotice = ref(null)
const lastRunSummary = ref('')
const runId = ref(0)

// SinaTools preload UX
const sinatoolsCardState = ref('') // '', 'lazy_not_loaded', 'loading', 'loaded', 'error'
const sinatoolsLoading = ref(false)
const sinatoolsProgressLabel = ref('')


const {
  toolStatuses,
  loading: statusLoading,
  error: statusError,
  refresh,
  toolStatus,
  toolReason,
} = useToolStatus()

const toolOptions = computed(() =>
  toolOrder(TOOL_KEYS).map((key) => ({
    key,
    ...TOOL_CONFIG[key],
  })),
)
const TOOL_SECTION_DEFS = [
  {
    key: 'morphology',
    kicker: 'Morphology and lexical evidence',
    title: 'Morphology / Lexical analyzers',
    description: 'Lemma, root, POS, and morphological evidence. These analyzers are inspected as lexical-morphology evidence, not as syntax parsers.',
    tools: ['camel', 'alkhalil', 'sinatools', 'madamira'],
    layout: 'layout-morphology',
  },
  {
    key: 'syntax',
    kicker: 'Syntax and UD evidence',
    title: 'Syntax / UD analyzers',
    description: 'POS and dependency-oriented evidence from UD-style pipelines.',
    tools: ['stanza', 'udpipe'],
    layout: 'layout-syntax',
  },
  {
    key: 'segmentation',
    kicker: 'Segmentation and lightweight lexical support',
    title: 'Segmentation / lexical support',
    description: 'Segmentation anchor and rule-based lexical support are shown separately because their capabilities are not identical to full morphology analyzers.',
    tools: ['farasa', 'qalsadi'],
    layout: 'layout-support',
  },
  {
    key: 'contextual',
    kicker: 'Contextual evidence',
    title: 'Contextual model output',
    description: 'Contextual representation models are not treated as direct morphology-table competitors. Missing morphology fields are N/A, not analyzer failure.',
    tools: ['arabert'],
    layout: 'layout-contextual',
  },
]

const groupedToolSections = computed(() =>
  TOOL_SECTION_DEFS
    .map((section) => ({
      ...section,
      tools: section.tools
        .filter((key) => TOOL_CONFIG[key])
        .map((key) => ({ key, ...TOOL_CONFIG[key] })),
    }))
    .filter((section) => section.tools.length > 0),
)

const toolStatusesLoaded = computed(() => Object.keys(toolStatuses.value).length > 0)
const tokenEstimate = computed(() => (inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0))
const selectedToolMeta = computed(() => TOOL_CONFIG[selectedTool.value] || { label: 'All tools', color: 'var(--c-text-primary)', type: 'Combined analysis' })
const TOOL_CAPABILITY_COPY = {
  all: { description: 'Combined analyzer evidence with expert fusion available as a separate research view.', features: ['Cross-tool evidence', 'Fusion trace', 'Conflicts'] },
  camel: { description: 'Strong lexical and morphological analysis for lemma, root, POS, and inflectional features.', features: ['Lemma', 'Root', 'Morphology', 'POS'] },
  farasa: { description: 'Segmentation-focused evidence for Arabic clitic and token boundary analysis.', features: ['Segmentation', 'Clitic boundaries'] },
  stanza: { description: 'UD-oriented contextual analysis with POS, lemma, morphology, and dependency parsing.', features: ['POS', 'Dependency', 'Lemma'] },
  udpipe: { description: 'Universal Dependencies pipeline used for POS and dependency-oriented supporting evidence.', features: ['UD POS', 'Dependency'] },
  qalsadi: { description: 'Lightweight Arabic lemmatization and rule-based supporting evidence.', features: ['Lemma support', 'Rule-based evidence'] },
  alkhalil: { description: 'Rule-based Arabic morphological analysis used as supporting morphological evidence.', features: ['Morphology', 'Root support', 'Lemma support'] },
  sinatools: { description: 'Lexical and lemma-oriented evidence when the local SinaTools resource is loaded.', features: ['Lemma', 'Lexical evidence'] },
  arabert: { description: 'Contextual representation support; not treated as a direct morphology analyzer.', features: ['Contextual support', 'Embeddings'] },
  madamira: { description: 'Context-sensitive morphological and disambiguation support when configured.', features: ['Morphology', 'Disambiguation'] },
}
const selectedCapability = computed(() => TOOL_CAPABILITY_COPY[selectedTool.value] || {
  description: selectedToolMeta.value.type || 'Analyzer-specific evidence.',
  features: TOOL_CONFIG[selectedTool.value]?.provides || [],
})
const averageConfidenceLabel = computed(() => {
  const scores = currentRows.value.map((row) => Number(row.confidence)).filter(Number.isFinite)
  return scores.length ? `${Math.round((scores.reduce((sum, score) => sum + score, 0) / scores.length) * 100)}%` : 'N/A'
})
const missingEvidenceCount = computed(() => currentRows.value.reduce((count, row) => {
  const values = Object.values(row.values || {})
  return count + values.filter((value) => !hasTokenValue(value)).length
}, 0))

const hasResults = computed(() => Boolean(rawResults.value || fusionPayload.value))
const jsonExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'json') : '#'))
const csvExportHref = computed(() => (hasResults.value ? exportUrl(inputText.value, 'csv') : '#'))
const prettyJson = computed(() => JSON.stringify({ analysis: rawResults.value, fusion: fusionPayload.value }, null, 2))
const fusionRows = computed(() => fusionPayload.value || [])
const allToolRows = computed(() =>
  TOOL_KEYS.flatMap((toolKey) =>
    toolRows(toolKey).map((row) => ({
      ...row,
      tool: toolKey,
    })),
  ),
)
const currentRows = computed(() => {
  if (selectedTool.value === 'all') {
    return allToolRows.value
  }
  return toolRows(selectedTool.value)
})
const tokenTimelineChart = computed(() => {
  const rows = currentRows.value
  const values = rows.map((row) => Math.round((Number(row.confidence) || 0) * 100))
  return {
    labels: rows.map((row) => `${row.index + 1}`),
    datasets: [
      {
        label: 'Confidence %',
        data: values,
        borderColor: '#4F46E5',
        backgroundColor: 'rgba(79, 70, 229, 0.14)',
      },
    ],
  }
})
const posDistributionChart = computed(() => {
  const counts = {}
  currentRows.value.forEach((row) => {
    const pos = String(row.values?.pos || '').trim()
    if (!pos) return
    counts[pos] = (counts[pos] || 0) + 1
  })
  const labels = Object.keys(counts)
  return {
    labels,
    datasets: [
      {
        label: 'Tokens',
        data: labels.map((label) => counts[label]),
        backgroundColor: ['#4F46E5', '#14B8A6', '#D97706', '#7C3AED', '#0EA5E9', '#22C55E'],
      },
    ],
  }
})
const morphologyChart = computed(() => {
  const counts = { Gender: 0, Number: 0, Tense: 0 }
  currentRows.value.forEach((row) => {
    if (row.values?.gender) counts.Gender += 1
    if (row.values?.number) counts.Number += 1
    if (row.values?.tense) counts.Tense += 1
  })
  const labels = Object.keys(counts).filter((label) => counts[label] > 0)
  return {
    labels,
    datasets: [
      {
        label: 'Morphology fields',
        data: labels.map((label) => counts[label]),
        backgroundColor: ['#7C3AED', '#D97706', '#14B8A6'],
      },
    ],
  }
})
const heatmapRows = computed(() => currentRows.value.map((row) => row.surface))
const heatmapCols = ['Lemma', 'Root', 'POS', 'Confidence']
const heatmapValues = computed(() =>
  currentRows.value.map((row) => [
    hasTokenValue(row.values?.lemma) ? 1 : 0,
    hasTokenValue(row.values?.root) ? 1 : 0,
    hasTokenValue(row.values?.pos) ? 1 : 0,
    Math.round((Number(row.confidence) || 0) * 100) / 100,
  ]),
)
const toolContributionChart = computed(() => {
  const counts = {}
  currentRows.value.forEach((row) => {
    Object.keys(row.values || {}).forEach((field) => {
      if (!hasTokenValue(row.values[field])) return
      const source = sourceForCurrentTool(field)
      if (!source) return
      counts[source] = (counts[source] || 0) + 1
    })
  })
  const labels = Object.keys(counts)
  return {
    labels,
    datasets: [
      {
        label: 'Evidence fields',
        data: labels.map((label) => counts[label]),
        backgroundColor: ['#4F46E5', '#14B8A6', '#D97706', '#7C3AED'],
      },
    ],
  }
})
const tabs = computed(() => [
  { key: 'results', label: selectedTool.value === 'all' ? 'All tools' : 'Token breakdown' },
  ...(selectedTool.value === 'all' ? [{ key: 'fusion', label: fusionLoading.value ? 'Fusion loading' : 'Fusion' }] : []),
  { key: 'json', label: 'JSON' },
])


const TOOL_TABLE_COLUMNS = {
  camel: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'root', label: 'Root' },
    { key: 'pos', label: 'POS' },
    { key: 'gender', label: 'Gender' },
    { key: 'number', label: 'Number' },
    { key: 'tense', label: 'Tense' },
    { key: 'gloss', label: 'Gloss' },
  ],
  alkhalil: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'root', label: 'Root' },
    { key: 'pos', label: 'Canonical POS' },
    { key: 'analysis', label: 'Morphological analysis', virtual: true },
  ],
  sinatools: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'pos', label: 'POS' },
    { key: 'root', label: 'Root' },
    { key: 'confidence', label: 'Confidence', virtual: true },
  ],
  madamira: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'pos', label: 'POS' },
    { key: 'gender', label: 'Gender' },
    { key: 'number', label: 'Number' },
    { key: 'tense', label: 'Tense' },
    { key: 'case', label: 'Case' },
    { key: 'confidence', label: 'Confidence', virtual: true },
  ],
  stanza: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'pos', label: 'UD POS' },
    { key: 'case', label: 'Case' },
    { key: 'deprel', label: 'Dependency relation' },
    { key: 'head', label: 'Head' },
  ],
  udpipe: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'pos', label: 'UD POS' },
    { key: 'deprel', label: 'Dependency relation' },
    { key: 'head', label: 'Head' },
  ],
  farasa: [
    { key: 'segmentation', label: 'Segmentation' },
    { key: 'evidence', label: 'Segmentation Evidence', virtual: true },
  ],
  qalsadi: [
    { key: 'lemma', label: 'Lemma' },
    { key: 'pos', label: 'POS', optional: true },
    { key: 'evidence', label: 'Lemmatization Evidence', virtual: true },
  ],
  arabert: [
    { key: 'contextual', label: 'Contextual Output', virtual: true },
    { key: 'evidence', label: 'Representation Status', virtual: true },
  ],
}

function toolTableColumns(toolKey) {
  const columns = TOOL_TABLE_COLUMNS[toolKey] || []
  return columns.filter((column) => {
    if (!column.optional) return true
    return toolRows(toolKey).some((row) => hasTokenValue(row.values?.[column.key]))
  })
}

function toolExpectedFields(toolKey) {
  const expected = {
    camel: ['lemma', 'root', 'pos'],
    alkhalil: ['lemma', 'root'],
    sinatools: ['lemma'],
    madamira: ['lemma', 'pos'],
    stanza: ['lemma', 'pos', 'dependency'],
    udpipe: ['lemma', 'pos', 'dependency'],
    farasa: ['segmentation'],
    qalsadi: ['lemma'],
    arabert: [],
  }

  return expected[toolKey] || []
}

function evidenceStatusForTool(toolKey, row) {
  if (toolKey === 'arabert') {
    return {
      label: 'Contextual representation',
      className: 'status-na',
    }
  }

  const expected = toolExpectedFields(toolKey)

  if (!expected.length) {
    return {
      label: 'Capability-specific output',
      className: 'status-na',
    }
  }

  const present = expected.filter((field) =>
    hasTokenValue(row?.values?.[field]),
  ).length

  if (present === expected.length) {
    return {
      label: 'Expected evidence present',
      className: 'evidence-complete',
    }
  }

  if (present === 0) {
    return {
      label: 'Expected evidence absent',
      className: 'evidence-missing',
    }
  }

  return {
    label: `Partial ${present}/${expected.length}`,
    className: 'evidence-partial',
  }
}

function toolCellValue(toolKey, field, row) {
  if (field === 'evidence') {
    return evidenceStatusForTool(toolKey, row).label
  }

  if (field === 'analysis') {
    return row.values?.analysis || 'Raw analysis not returned'
  }

  if (field === 'contextual') {
    const contextual =
      row?.values?.embedding ||
      row?.values?.contextual ||
      row?.values?.representation ||
      row?.values?.embedding_status

    if (hasTokenValue(contextual)) {
      if (typeof contextual === 'string') return contextual
      return 'Contextual representation available'
    }

    return 'Contextual model only - no direct morphology labels'
  }

  return displayTokenValue(toolKey, field, row?.values?.[field], resultStatus(toolKey))
}

function toolCellClass(toolKey, field, row) {
  if (field === 'pos') return 'pos-compact'
  if (field === 'evidence') {
    return [
      'evidence-status',
      evidenceStatusForTool(toolKey, row).className,
    ]
  }
  if (['lemma', 'root', 'stem', 'segmentation'].includes(field)) {
    return 'arabic-value compact-primary-value'
  }
  if (['dependency', 'deprel', 'head'].includes(field)) return 'dependency-value'
  if (field === 'contextual') return 'contextual-value'
  return 'compact-secondary'
}

function toolCellDirection(field) {
  if (['lemma', 'root', 'stem', 'segmentation'].includes(field)) return 'rtl'
  return null
}

function evidenceStatus(row) {
  if (selectedTool.value === 'arabert') {
    return { label: 'Contextual output', className: 'status-na' }
  }
  const expected = selectedTool.value === 'all'
    ? Object.keys(row.values || {})
    : (TOOL_CONFIG[selectedTool.value]?.provides || [])
  if (!expected.length) return { label: 'N/A for direct fields', className: 'status-na' }
  const present = expected.filter((field) => hasTokenValue(row.values?.[field])).length
  if (present === expected.length) return { label: 'Expected evidence present', className: 'evidence-complete' }
  if (present === 0) return { label: 'Expected evidence absent', className: 'evidence-missing' }
  return { label: `Partial ${present}/${expected.length}`, className: 'evidence-partial' }
}

function selectTool(toolKey) {
  selectionNotice.value = null
  pendingTool.value = ''

  if (toolKey === 'all') {
    selectedTool.value = 'all'
    return
  }

  const status = toolStatus(toolKey)
  const display = statusDisplay(status)
  if (['available', 'partial', 'lazy', 'loading', 'unknown'].includes(display.group)) {
    selectedTool.value = toolKey
    selectionNotice.value =
      display.group === 'lazy'
        ? {
            kind: 'info',
            title: `${TOOL_CONFIG[toolKey].label} loads on demand`,
            message: 'This local resource is lazy and is not loaded by normal analysis pages.',
            pending: false,
          }
        : null
    return
  }

  pendingTool.value = toolKey
  selectionNotice.value = {
    kind: 'warn',
    title: `${TOOL_CONFIG[toolKey].label} is currently ${statusLabel(status)}`,
    message: toolReason(toolKey) || 'You can still choose it, but the backend is reporting a non-active state.',
    pending: true,
  }
}

function confirmPendingSelection() {
  if (!pendingTool.value) return
  selectedTool.value = pendingTool.value
  selectionNotice.value = null
  pendingTool.value = ''
}

function cancelPendingSelection() {
  pendingTool.value = ''
  selectionNotice.value = null
}

async function analyze() {
  if (!inputText.value.trim()) return
  if (statusLoading.value && !toolStatusesLoaded.value) await refresh()

  const currentRunId = ++runId.value
  loading.value = true
  fusionLoading.value = false
  error.value = ''
  fusionError.value = ''
  rawResults.value = null
  fusionPayload.value = null
  copied.value = false

  try {
    const data =
      selectedTool.value === 'all'
        ? await analyzeAll(inputText.value)
        : await analyzeTool(selectedTool.value, inputText.value)

    if (currentRunId !== runId.value) return
    rawResults.value = data
    if (selectedTool.value === 'all') {
      loadFusion(currentRunId)
    }
    captureRunSummary()
  } catch (e) {
    if (currentRunId === runId.value) error.value = readError(e, 'Failed to connect to the backend.')
  } finally {
    if (currentRunId === runId.value) loading.value = false
  }
}

async function loadFusion(currentRunId = runId.value) {
  if (!inputText.value.trim()) return
  fusionLoading.value = true
  fusionError.value = ''
  try {
    const data = await fusionText(inputText.value)
    if (currentRunId !== runId.value) return
    fusionPayload.value = normalizeFusionRows(data)
  } catch (e) {
    if (currentRunId !== runId.value) return
    fusionPayload.value = []
    fusionError.value = e?.message || 'Detailed fusion payload was not returned.'
  } finally {
    if (currentRunId === runId.value) fusionLoading.value = false
  }
}

function toolRows(toolKey) {
  const payload = toolPayloadForKey(toolKey)
  const tokens = Array.isArray(payload?.tokens) ? payload.tokens : []
  const fields = (TOOL_TABLE_COLUMNS[toolKey] || [])
    .filter((column) => !column.virtual)
    .map((column) => column.key)

  return tokens.map((token, index) => {
    const best = canonicalToken(token)
    const values = {}

    for (const field of fields) {
      values[field] = readField(best, field)
    }

    if (toolKey === 'alkhalil') {
      values.analysis = summarizeAnalysis(best)
    }

    return {
      index,
      surface: token.surface || token.word || `#${index + 1}`,
      values,
      confidence: confidenceFromToken(best),
    }
  })
}

function toolColumns(toolKey) {
  return TOOL_CONFIG[toolKey].provides
}

function normalizeFusionRows(payload) {
  if (!payload) return []
  const source = payload?.data || payload
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(source?.fusion)
      ? source.fusion
      : Array.isArray(source?.fusion_result?.fusion)
        ? source.fusion_result.fusion
        : Array.isArray(source?.result)
          ? source.result
          : []

  return rows.map((row, index) => ({
    index,
    word: row?.word || row?.surface || row?.token || `#${index + 1}`,
    final: normalizeFinal(row),
    sources: row?.sources || row?.fusion?.chosen_sources || {},
    conflicts: Array.isArray(row?.conflicts) ? row.conflicts : [],
    notes: Array.isArray(row?.notes) ? row.notes : [],
    decision_trace: Array.isArray(row?.decision_trace) ? row.decision_trace : [],
  }))
}

function normalizeFinal(row) {
  const final = { ...(row?.final || {}) }
  const legacyFusion = row?.fusion || {}
  if (!final.pos && legacyFusion.final_pos) final.pos = legacyFusion.final_pos
  if (!final.lemma && legacyFusion.final_lemma) final.lemma = legacyFusion.final_lemma
  if (!final.segmentation && legacyFusion.final_segmentation) final.segmentation = legacyFusion.final_segmentation
  if (!final.confidence_level && legacyFusion.confidence_level) final.confidence_level = legacyFusion.confidence_level
  if (final.confidence_score === undefined && legacyFusion.confidence !== undefined) final.confidence_score = legacyFusion.confidence
  return final
}

function toolPayloadForKey(toolKey) {
  const payload = analysisPayload()
  if (selectedTool.value === 'all') {
    return payload?.tools?.[toolKey] ?? payload?.[toolKey] ?? null
  }
  return payload
}

function resultStatus(toolKey) {
  return toolPayloadForKey(toolKey)?.status || toolStatus(toolKey)
}

function resultReason(toolKey) {
  return toolPayloadForKey(toolKey)?.reason || toolReason(toolKey)
}

function analysisPayload() {
  return rawResults.value?.data || rawResults.value
}

function fusionSourceChips(sources) {
  return Object.entries(sources || {}).map(([feature, tool]) => ({
    label: `${feature}: ${TOOL_CONFIG[tool]?.label || tool}`,
    color: TOOL_CONFIG[tool]?.color || 'var(--c-text-secondary)',
  }))
}

function unwrapAnalysis(token) {
  return canonicalToken(token)
}

function formatSegmentation(token) {
  if (Array.isArray(token?.segmentation) && token.segmentation.length) return token.segmentation.join(' + ')
  if (Array.isArray(token?.segments) && token.segments.length) return token.segments.join(' + ')
  if (Array.isArray(token?.parts) && token.parts.length) return token.parts.join(' + ')
  return token?.segmentation || token?.segments || token?.parts || ''
}

function confidenceFromToken(token) {
  const raw = token?.confidence_level || token?.confidence?.level || token?.confidence
  if (raw) return String(raw).toLowerCase()
  const score = token?.confidence_score ?? token?.confidence?.score
  if (typeof score === 'number') {
    if (score >= 0.75) return 'high'
    if (score >= 0.45) return 'medium'
    return 'low'
  }
  return ''
}

function readField(raw, field) {
  if (!raw) return ''

  if (field === 'pos') return raw.pos || raw.upos || ''
  if (field === 'confidence_score') return raw.confidence?.score ?? raw.confidence_score ?? ''
  if (field === 'confidence_level') return raw.confidence?.level ?? raw.confidence_level ?? ''

  if (field === 'segmentation') {
    const arr =
      raw.segmentation ||
      raw.segments ||
      raw.parts

    if (Array.isArray(arr)) return arr.join(' + ')
    return arr || ''
  }

  if (field === 'dependency') {
    const dependency = raw.dependency || raw.dep || {}
    const deprel = dependency.deprel || raw.deprel || ''
    return deprel ? (dependency.head_text ? `${deprel} -> ${dependency.head_text}` : deprel) : ''
  }

  if (field === 'deprel') {
    const dependency = raw.dependency || raw.dep || {}
    return dependency.deprel || raw.deprel || ''
  }

  if (field === 'head') {
    const dependency = raw.dependency || raw.dep || {}
    return dependency.head_text || dependency.head || raw.head || ''
  }

  return raw[field] || ''
}

function summarizeAnalysis(raw) {
  const analysis = Array.isArray(raw?.analyses) ? raw.analyses[0] : null
  if (!analysis || typeof analysis !== 'object') return ''
  return analysis.type || analysis.gloss || analysis.pos || ''
}


function compactValue(row, field) {
  return displayTokenValue(field, row?.values?.[field])
}

function compactRootOrSeg(row) {
  if (hasTokenValue(row?.values?.root)) return displayTokenValue('root', row.values.root)
  if (hasTokenValue(row?.values?.segmentation)) return displayTokenValue('segmentation', row.values.segmentation)
  if (hasTokenValue(row?.values?.dependency)) return displayTokenValue('dependency', row.values.dependency)
  return 'N/A'
}

function compactRootIsArabic(row) {
  return hasTokenValue(row?.values?.root) || hasTokenValue(row?.values?.segmentation)
}

function compactRootClass(row) {
  return compactRootIsArabic(row) ? 'arabic-value compact-secondary' : 'compact-secondary'
}

function fieldLabel(field) {
  if (field === 'pos') return 'POS'
  if (field === 'segmentation') return 'Segmentation'
  if (field === 'dependency') return 'Dependency'
  if (field === 'confidence_score') return 'Confidence score'
  if (field === 'confidence_level') return 'Confidence level'
  if (field === 'case') return 'Case'
  if (field === 'definite') return 'Definite'
  if (field === 'gender') return 'Gender'
  if (field === 'number') return 'Number'
  if (field === 'tense') return 'Tense'
  if (field === 'gloss') return 'Gloss'
  if (field === 'stem') return 'Stem'
  if (field === 'lemma') return 'Lemma'
  if (field === 'root') return 'Root'
  return field
}

function isArabicField(field) {
  return ['lemma', 'root', 'gloss', 'stem'].includes(field)
}

function formatCellValue(field, value) {
  if (!hasTokenValue(value)) return missingFieldLabel(selectedTool.value, field, resultStatus(selectedTool.value))
  if (Array.isArray(value)) return value.join(' + ')

  if (field === 'segmentation') {
    if (Array.isArray(value)) return value.join(' + ')
    if (!value) return missingFieldLabel(selectedTool.value, field, resultStatus(selectedTool.value))
    return String(value)
  }

  return String(value)
}

function cellClass(field, value) {
  return !hasTokenValue(value)
    ? 'field-value muted not-recognized'
    : isArabicField(field)
      ? 'field-value arabic'
      : 'field-value'
}

function statusBadge(toolKey) {
  const display = statusDisplay(toolStatus(toolKey), toolReason(toolKey))
  if (display.status === 'ok' || display.status === 'loaded') return { show: false, label: '', className: '' }
  return { show: true, label: display.label, className: display.className }
}

function statusLabel(status) {
  return statusDisplay(status).label
}

function isRenderableToolStatus(status) {
  return canRenderToolEvidence(status)
}

function statusPill(status) {
  return statusDisplay(status).className
}

function confidencePill(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'high') return 'pill-green'
  if (normalized === 'medium') return 'pill-blue'
  if (normalized === 'low') return 'pill-red'
  return 'pill-gray'
}

function severityPill(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'high') return 'pill-red'
  if (normalized === 'medium') return 'pill-amber'
  if (normalized === 'low') return 'pill-yellow'
  return 'pill-gray'
}

function missingFieldLabel(toolKey, field, status = 'ok') {
  return missingValueLabel(toolKey, field, status)
}

function displayTokenValue(toolOrField, fieldOrValue, maybeValue, maybeStatus) {
  const usingLegacySignature = maybeValue === undefined
  const toolKey = usingLegacySignature ? selectedTool.value : toolOrField
  const field = usingLegacySignature ? toolOrField : fieldOrValue
  const value = usingLegacySignature ? fieldOrValue : maybeValue
  const status = maybeStatus || resultStatus(toolKey)
  if (Array.isArray(value)) {
    return value.length ? value.join(' + ') : missingFieldLabel(toolKey, field, status)
  }
  if (!hasTokenValue(value)) {
    return missingFieldLabel(toolKey, field, status)
  }
  return String(value)
}

function displayConflictValue(value) {
  if (!hasTokenValue(value)) return 'Not returned'
  return String(value)
}

function statusDescription(status) {
  return statusDisplay(status).label
}

function emptyToolOutputLabel(toolKey) {
  const status = resultStatus(toolKey)
  if (toolKey === 'arabert') return 'Contextual metadata was not returned for this run.'
  if (toolKey === 'madamira') return 'MADAMIRA is excluded - missing licensed resources.'
  return missingValueLabel(toolKey, '*', status)
}

function hasTokenValue(value) {
  if (Array.isArray(value)) return value.length > 0
  return value !== null && value !== undefined && value !== '' && value !== 'null'
}

function metricPercent(value) {
  const parsed = Number.parseFloat(String(value || '').replace('%', ''))
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0
}

function formatDecimal(value) {
  return typeof value === 'number' ? value.toFixed(3) : '0.000'
}

function readError(errorObject, fallback) {
  return errorObject?.response?.data?.detail || errorObject?.response?.data?.error || errorObject?.message || fallback
}

function loadSample() {
  inputText.value = 'وجدت المعلمة طالبة مجتهدة في الفصل'
}

function clear() {
  inputText.value = ''
  rawResults.value = null
  fusionPayload.value = null
  error.value = ''
  copied.value = false
  selectionNotice.value = null
  pendingTool.value = ''
  selectedTool.value = 'all'
  activeTab.value = 'results'
  lastRunSummary.value = ''
}

function guardExport(event) {
  if (!hasResults.value) event.preventDefault()
}

async function copyCurrentJson() {
  if (!hasResults.value) return
  await navigator.clipboard.writeText(prettyJson.value)
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}

onMounted(async () => {
  if (route.query.tool && TOOL_CONFIG[String(route.query.tool)]) {
    selectedTool.value = String(route.query.tool)
  }
  if (route.query.text) {
    inputText.value = String(route.query.text)
  }
  await nextTick()
  if (route.query.text) {
    analyze()
  }
})

function sourceForCurrentTool(field) {
  if (selectedTool.value === 'all') {
    return field === 'segmentation' ? 'Farasa' : field === 'dependency' ? 'Stanza' : 'Fusion'
  }
  return TOOL_CONFIG[selectedTool.value]?.label || selectedTool.value
}

function captureRunSummary() {
  const toolLabel = selectedTool.value === 'all' ? 'All tools' : TOOL_CONFIG[selectedTool.value]?.label || selectedTool.value
  const rows = currentRows.value
  const tokenCount = rows.length
  const avgConfidence = rows.length
    ? Math.round((rows.reduce((sum, row) => sum + (Number(row.confidence) || 0), 0) / rows.length) * 100)
    : 0
  lastRunSummary.value = `${toolLabel} | ${tokenCount} tokens | ${avgConfidence}% average confidence`
  recordAnalysis({
    page: 'Analyze',
    text: inputText.value.trim(),
    summary: lastRunSummary.value,
  })
}
</script>

<style scoped>
.analyze-page {
  width: min(96vw, 1500px);
}

.analyze-hero {
  min-height: 250px;
}

.compact-actions,
.run-row {
  margin-top: 0;
}

.run-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
}

.disabled {
  pointer-events: none;
  opacity: 0.5;
}

.copy-note {
  color: var(--green);
  font-size: 13px;
  font-weight: 500;
}

.selector-panel {
  margin-top: 18px;
}

.selector-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.selector-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  background: var(--c-surface);
  text-align: left;
  cursor: pointer;
  transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}

.selector-card:hover,
.selector-card.active {
  transform: translateY(-2px);
  border-color: var(--c-accent-border);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}

.selector-card.all-tools {
  grid-column: 1 / -1;
  background: linear-gradient(135deg, var(--c-page-bg) 0%, var(--c-accent-light) 100%);
}

.selector-dot {
  width: 12px;
  height: 12px;
  margin-top: 4px;
  border-radius: 999px;
  flex: 0 0 auto;
}

.selector-dot--neutral {
  background: var(--c-text-primary);
}

.selector-copy {
  display: grid;
  gap: 4px;
}

.selector-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.selector-label {
  color: var(--c-text-primary);
  font-size: 15px;
  font-weight: 500;
}

.selector-subtitle {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}

.selection-notice {
  display: grid;
  gap: 8px;
  margin-top: 14px;
  padding: 14px;
  border-radius: 10px;
}

.selection-notice.info {
  border: 1px solid var(--c-accent-border);
  background: var(--c-accent-light);
  color: var(--c-accent-text);
}

.selection-notice.warn {
  border: 1px solid var(--c-conf-med-border);
  background: var(--c-conf-med-bg);
  color: var(--c-conf-med-text);
}

.analysis-loading {
  min-height: 170px;
}

.info-state {
  padding: 12px 14px;
  border: 1px solid var(--c-accent-border);
  border-radius: 10px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 13px;
  line-height: 1.55;
}

.info-state--warn {
  border-color: var(--c-warning-border);
  background: var(--c-warning-bg);
  color: var(--c-warning-text);
}

.loading-stack {
  width: min(520px, 100%);
  display: grid;
  gap: 12px;
}

.loading-stack .wide {
  width: 78%;
}

.loading-stack .short {
  width: 42%;
}

.analysis-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 20px;
  padding: 5px;
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  background: var(--c-page-bg);
}

.analysis-tabs button {
  min-height: 38px;
  padding: 8px 14px;
  border-radius: var(--radius-control);
  color: var(--muted);
  background: transparent;
  cursor: pointer;
  font-weight: 500;
}

.analysis-tabs button.active {
  color: var(--c-accent-text);
  background: var(--c-accent-light);
  box-shadow: 0 1px 7px rgba(23, 32, 51, 0.09);
}

.tab-panel {
  display: grid;
  gap: 18px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.tool-result-card,
.single-result-card {
  display: grid;
  gap: 14px;
}

.tool-result-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  min-width: 0;
  padding-top: 10px;
  border-top: 4px solid transparent;
}

.tool-result-title {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  min-width: 0;
  flex: 1 1 190px;
}

.tool-result-head .pill {
  max-width: 100%;
  white-space: normal;
  text-align: center;
}

.tool-result-bar {
  width: 12px;
  height: 42px;
  border-radius: 999px;
  flex: 0 0 auto;
}

.tool-result-title h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 500;
}

.tool-result-title p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 13px;
  font-weight: 500;
}

.tool-table {
  max-height: 420px;
}

.status-card {
  max-width: 100%;
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-page-bg);
}

.status-card strong {
  display: block;
  overflow-wrap: anywhere;
  font-weight: 500;
}

.status-card p {
  margin: 6px 0 0;
  overflow-wrap: anywhere;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}

.fusion-table {
  max-height: 520px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.source-chip {
  padding: 6px 10px;
  border-radius: 999px;
  color: white;
  font-size: 12px;
  font-weight: 500;
}

.field-value {
  color: var(--ink);
  font-size: 14px;
  font-weight: 500;
}

.field-value.arabic {
  font-size: 16px;
}

.field-value.muted {
  color: var(--muted);
}

.not-recognized {
  color: var(--c-text-muted);
  font-style: italic;
  font-size: 0.8rem;
}

.evaluation-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.metric-block {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.metric-block-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.metric-block strong {
  color: var(--c-text-primary);
  font-size: 20px;
  font-weight: 500;
}

.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metrics-note {
  margin: 14px 0 0;
  color: var(--muted);
}

.metric-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 60px;
  min-height: 34px;
  padding: 6px 10px;
  border-radius: var(--radius-control);
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 15px;
  font-weight: 500;
}

.tool-chip-active {
  background: var(--c-conf-high-border);
}

.tool-chip-muted {
  background: var(--c-text-muted);
}

.analysis-visual-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.analysis-visual-grid--wide {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.selected-capability-panel {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  padding: 18px 20px;
  border: 1px solid var(--c-border);
  border-left: 4px solid var(--c-accent);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}
.selected-capability-panel h2 { margin: 5px 0 7px; font-size: 1.15rem; }
.selected-capability-panel p { margin: 0; color: var(--c-text-secondary); line-height: 1.55; max-width: 760px; }
.capability-tags { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; }
.capability-tags span { padding: 6px 9px; border-radius: 999px; background: var(--c-accent-light); color: var(--c-accent); font-size: .76rem; font-weight: 700; }
.analysis-summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.analysis-summary-card { display: grid; gap: 6px; padding: 15px; border: 1px solid var(--c-border); border-radius: var(--radius-card); background: var(--c-surface); }
.analysis-summary-card span { color: var(--c-text-secondary); font-size: .78rem; }
.analysis-summary-card strong { font-size: 1.2rem; }
.evidence-status { display: inline-flex; padding: 5px 8px; border-radius: 999px; font-size: .72rem; font-weight: 700; white-space: nowrap; }
.evidence-complete { color: #166534; background: #dcfce7; }
.evidence-partial { color: #92400e; background: #fef3c7; }
.evidence-missing { color: #64748b; background: #f1f5f9; }

@media (max-width: 1100px) {
  .analysis-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .selector-grid,
  .results-grid,
  .evaluation-grid,
  .analysis-visual-grid,
  .analysis-visual-grid--wide {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .selected-capability-panel { align-items: flex-start; flex-direction: column; }
  .capability-tags { justify-content: flex-start; }
  .analysis-summary-grid { grid-template-columns: 1fr 1fr; }
  .selector-grid,
  .results-grid,
  .evaluation-grid,
  .analysis-visual-grid,
  .analysis-visual-grid--wide {
    grid-template-columns: 1fr;
  }

  .metric-main {
    align-items: flex-start;
  }
}


.compact-evidence-table {
  overflow: hidden;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-card);
  background: var(--c-surface);
}

.compact-evidence-table table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
}

.compact-evidence-table th,
.compact-evidence-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #edf1f6;
  vertical-align: middle;
}

.compact-evidence-table th {
  position: static;
  background: var(--c-page-bg);
  color: var(--c-text-secondary);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.compact-evidence-table th:nth-child(1) { width: 18%; }
.compact-evidence-table th:nth-child(2) { width: 18%; }
.compact-evidence-table th:nth-child(3) { width: 12%; }
.compact-evidence-table th:nth-child(4) { width: 22%; }
.compact-evidence-table th:nth-child(5) { width: 18%; }
.compact-evidence-table th:nth-child(6) { width: 12%; }

.compact-evidence-table td {
  color: var(--c-text-primary);
  font-size: 13px;
  overflow-wrap: anywhere;
}

.token-cell {
  font-size: 15px;
  font-weight: 700;
}

.pos-compact {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: 11px;
  font-weight: 700;
}

.compact-secondary {
  color: var(--c-text-secondary);
  font-size: 13px;
}

@media (max-width: 760px) {
  .compact-evidence-table table,
  .compact-evidence-table thead,
  .compact-evidence-table tbody,
  .compact-evidence-table tr,
  .compact-evidence-table th,
  .compact-evidence-table td {
    display: block;
  }

  .compact-evidence-table thead {
    display: none;
  }

  .compact-evidence-table tr {
    padding: 12px;
    border-bottom: 1px solid var(--c-border);
  }

  .compact-evidence-table td {
    display: grid;
    grid-template-columns: 120px minmax(0, 1fr);
    gap: 10px;
    padding: 7px 0;
    border-bottom: 0;
  }

  .compact-evidence-table td::before {
    color: var(--c-text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
  }

  .compact-evidence-table td:nth-child(1)::before { content: "Token"; }
  .compact-evidence-table td:nth-child(2)::before { content: "Lemma"; }
  .compact-evidence-table td:nth-child(3)::before { content: "POS"; }
  .compact-evidence-table td:nth-child(4)::before { content: "Root / Seg"; }
  .compact-evidence-table td:nth-child(5)::before { content: "Evidence"; }
  .compact-evidence-table td:nth-child(6)::before { content: "Confidence"; }
}


.capability-results {
  display: grid;
  gap: 22px;
}

.capability-group {
  display: grid;
  gap: 12px;
}

.capability-group-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  padding: 14px 16px;
  border: 1px solid var(--c-border);
  border-left: 4px solid var(--c-accent);
  border-radius: var(--radius-card);
  background: var(--c-page-bg);
}

.capability-group-head h3 {
  margin: 4px 0 4px;
  font-size: 1.15rem;
  font-weight: 650;
}

.capability-group-head p {
  max-width: 880px;
  margin: 0;
  color: var(--c-text-secondary);
  line-height: 1.55;
}

.group-count {
  flex: 0 0 auto;
  padding: 7px 10px;
  border-radius: 999px;
  background: var(--c-accent-light);
  color: var(--c-accent-text);
  font-size: .78rem;
  font-weight: 700;
}

.capability-tool-grid {
  display: grid;
  gap: 16px;
}

.capability-tool-grid.layout-morphology {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.capability-tool-grid.layout-syntax,
.capability-tool-grid.layout-support {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.capability-tool-grid.layout-contextual {
  grid-template-columns: minmax(0, 1fr);
}

.capability-tool-grid.layout-contextual .tool-result-card {
  max-width: 760px;
}

.compact-evidence-table th {
  font-size: 11px;
}

.compact-evidence-table td {
  font-size: 13.5px;
}

.compact-evidence-table th,
.compact-evidence-table td {
  padding: 11px 13px;
}

.token-cell {
  font-size: 16px;
}

@media (min-width: 1350px) {
  .capability-tool-grid.layout-morphology {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .capability-tool-grid.layout-syntax,
  .capability-tool-grid.layout-support {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 1100px) {
  .analyze-page {
    width: min(100% - 28px, 1240px);
  }

  .capability-tool-grid.layout-morphology,
  .capability-tool-grid.layout-syntax,
  .capability-tool-grid.layout-support,
  .capability-tool-grid.layout-contextual {
    grid-template-columns: 1fr;
  }

  .capability-group-head {
    align-items: flex-start;
    flex-direction: column;
  }
}


/* FINAL ANALYZE RESEARCH LAYOUT */
.analyze-page {
  width: min(94vw, 1380px);
}

.capability-results {
  gap: 28px;
}

.capability-group {
  gap: 12px;
}

.capability-group-head {
  padding: 14px 18px;
  align-items: center;
}

.capability-group-head h3 {
  font-size: 1.08rem;
}

.capability-group-head p {
  max-width: 920px;
  font-size: .84rem;
}

.capability-tool-grid,
.capability-tool-grid.layout-morphology,
.capability-tool-grid.layout-syntax,
.capability-tool-grid.layout-support {
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  align-items: start;
}

.capability-tool-grid.layout-contextual {
  grid-template-columns: 1fr !important;
}

.capability-tool-grid.layout-contextual .tool-result-card {
  max-width: none;
}

.tool-result-card {
  align-self: start;
  height: auto !important;
  min-height: 0 !important;
}

.tool-result-card .tool-card-header {
  min-height: 58px;
}

.compact-evidence-table {
  width: 100%;
}

.compact-evidence-table table {
  table-layout: auto;
}

.compact-evidence-table th,
.compact-evidence-table td {
  padding: 10px 9px;
}

.compact-evidence-table th {
  font-size: 10px;
  white-space: nowrap;
}

.compact-evidence-table td {
  font-size: 12.5px;
}

.compact-evidence-table th:nth-child(1),
.compact-evidence-table th:nth-child(2),
.compact-evidence-table th:nth-child(3),
.compact-evidence-table th:nth-child(4),
.compact-evidence-table th:nth-child(5),
.compact-evidence-table th:nth-child(6) {
  width: auto;
}

.token-cell {
  min-width: 64px;
  font-size: 14px;
}

.pos-compact {
  padding: 3px 7px;
  font-size: 10px;
}

.evidence-status,
.compact-evidence-table .pill {
  white-space: nowrap;
}

@media (max-width: 1050px) {
  .analyze-page {
    width: min(100% - 24px, 960px);
  }

  .capability-tool-grid,
  .capability-tool-grid.layout-morphology,
  .capability-tool-grid.layout-syntax,
  .capability-tool-grid.layout-support,
  .capability-tool-grid.layout-contextual {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 760px) {
  .capability-group-head {
    align-items: flex-start;
  }

  .compact-evidence-table td {
    grid-template-columns: 108px minmax(0, 1fr);
  }
}


/* Capability-specific analyzer tables */
.capability-specific-table table {
  table-layout: auto;
}

.capability-specific-table th,
.capability-specific-table td {
  padding: 10px 12px;
}

.capability-specific-table th {
  white-space: nowrap;
}

.compact-primary-value {
  display: inline-block;
  min-width: 24px;
}

.dependency-value,
.contextual-value {
  color: var(--c-text-secondary);
  font-size: 12px;
  line-height: 1.45;
}

.capability-specific-table .pos-compact {
  background: var(--c-accent-light);
  color: var(--c-accent-text);
}

@media (max-width: 760px) {
  .capability-specific-table table,
  .capability-specific-table thead,
  .capability-specific-table tbody,
  .capability-specific-table tr,
  .capability-specific-table th,
  .capability-specific-table td {
    display: block;
  }

  .capability-specific-table thead {
    display: none;
  }

  .capability-specific-table tr {
    padding: 12px;
    border-bottom: 1px solid var(--c-border);
  }

  .capability-specific-table td {
    display: grid;
    grid-template-columns: 118px minmax(0, 1fr);
    gap: 10px;
    padding: 7px 0;
    border-bottom: 0;
  }

  .capability-specific-table td::before {
    content: attr(data-label);
    color: var(--c-text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
  }

  .capability-specific-table td:first-child::before {
    content: "Token";
  }
}

</style>
