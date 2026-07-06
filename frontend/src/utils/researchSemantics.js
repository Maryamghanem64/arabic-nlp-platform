import { TOOL_CONFIG } from '@/config/tools'

export const STATUS_GROUPS = {
  available: new Set(['ok', 'loaded']),
  partial: new Set(['partial']),
  lazy: new Set(['lazy', 'lazy_not_loaded']),
  loading: new Set(['loading']),
  excluded: new Set(['excluded', 'disabled', 'future_work']),
  unavailable: new Set(['unavailable', 'missing_resources', 'missing_dependency', 'missing_model', 'missing_java', 'skipped_low_memory']),
  error: new Set(['error', 'timeout']),
  degraded: new Set(['degraded']),
}

export const TOOL_ROLES = {
  camel: 'Morphology, lemma, root, POS, gloss',
  farasa: 'Segmentation specialist',
  stanza: 'UD POS, lemma, and dependency syntax',
  qalsadi: 'Lemma-oriented lexical/rule-based support',
  alkhalil: 'Arabic morphology, root, lemma, canonical POS evidence',
  udpipe: 'UD POS, lemma, and dependency syntax',
  sinatools: 'Lazy local lexical resource',
  arabert: 'Contextual transformer support only',
  madamira: 'Excluded licensed-resource analyzer',
}

export const TOOL_CAPABILITY_NOTES = {
  arabert: {
    lemma: 'Not supported - contextual model',
    root: 'Not supported - contextual model',
    pos: 'Not supported - base model has no task-specific POS head',
    segmentation: 'Not supported - contextual model',
    dependency: 'Not supported - contextual model',
  },
  farasa: {
    lemma: 'Not supported - segmentation-focused tool',
    root: 'Not supported - segmentation-focused tool',
    pos: 'Not supported - segmentation-focused tool',
    dependency: 'Not supported - segmentation-focused tool',
  },
  stanza: {
    root: 'Not supported',
    segmentation: 'Not supported',
  },
  udpipe: {
    root: 'Not supported',
    segmentation: 'Not supported',
  },
  qalsadi: {
    root: 'Not supported',
    dependency: 'Not supported',
    segmentation: 'Not supported',
  },
  madamira: {
    '*': 'Excluded - missing licensed resources',
  },
  sinatools: {
    dependency: 'Not supported',
    segmentation: 'Not supported',
  },
}

export function normalizeStatus(status) {
  return String(status || 'unknown').trim().toLowerCase()
}

export function classifyToolStatus(status) {
  const normalized = normalizeStatus(status)
  for (const [group, statuses] of Object.entries(STATUS_GROUPS)) {
    if (statuses.has(normalized)) return group
  }
  return 'unknown'
}

export function isToolAvailable(status) {
  return ['available', 'partial'].includes(classifyToolStatus(status))
}

export function canRenderToolEvidence(status) {
  return ['available', 'partial', 'degraded', 'unknown'].includes(classifyToolStatus(status))
}

export function statusDisplay(status, reason = '') {
  const normalized = normalizeStatus(status)
  const displays = {
    ok: ['Available', 'pill-green'],
    loaded: ['Available', 'pill-green'],
    partial: ['Partial evidence', 'pill-amber'],
    lazy: ['Lazy local resource - not loaded', 'pill-blue'],
    lazy_not_loaded: ['Lazy local resource - not loaded', 'pill-blue'],
    loading: ['Loading local resource', 'pill-blue'],
    excluded: ['Excluded from current configuration', 'pill-gray'],
    disabled: ['Disabled', 'pill-gray'],
    unavailable: ['Unavailable', 'pill-gray'],
    timeout: ['Runtime timeout', 'pill-red'],
    error: ['Analyzer error', 'pill-red'],
    missing_resources: ['Required resources missing', 'pill-gray'],
    missing_dependency: ['Dependency missing', 'pill-gray'],
    missing_model: ['Model missing', 'pill-gray'],
    missing_java: ['Java runtime/resource unavailable', 'pill-gray'],
    skipped_low_memory: ['Skipped because of memory guard', 'pill-gray'],
    future_work: ['Future integration', 'pill-gray'],
    degraded: ['Degraded runtime evidence', 'pill-amber'],
    unknown: ['Status unknown', 'pill-gray'],
  }
  const [label, className] = displays[normalized] || displays.unknown
  return {
    status: normalized,
    group: classifyToolStatus(normalized),
    label,
    className,
    reason: reason || '',
  }
}

export function statusGroupsFromMap(statusMap = {}) {
  const groups = {
    activeTools: [],
    partialTools: [],
    lazyTools: [],
    loadingTools: [],
    excludedTools: [],
    unavailableTools: [],
    errorTools: [],
    degradedTools: [],
  }

  Object.entries(statusMap).forEach(([tool, entry]) => {
    const group = classifyToolStatus(entry?.status || entry)
    if (group === 'available') groups.activeTools.push(tool)
    else if (group === 'partial') groups.partialTools.push(tool)
    else if (group === 'lazy') groups.lazyTools.push(tool)
    else if (group === 'loading') groups.loadingTools.push(tool)
    else if (group === 'excluded') groups.excludedTools.push(tool)
    else if (group === 'unavailable') groups.unavailableTools.push(tool)
    else if (group === 'error') groups.errorTools.push(tool)
    else if (group === 'degraded') groups.degradedTools.push(tool)
  })

  return groups
}

export function toolRole(tool) {
  return TOOL_ROLES[tool] || TOOL_CONFIG[tool]?.researchRole || 'Analyzer evidence'
}

export function missingValueLabel(tool, field, status = 'ok') {
  const display = statusDisplay(status)
  if (display.group === 'lazy') return 'Lazy local resource - not loaded'
  if (display.group === 'loading') return 'Loading local resource'
  if (display.group === 'excluded') return tool === 'madamira' ? 'Excluded - missing licensed resources' : display.label
  if (display.group === 'unavailable' || display.group === 'error') return display.label

  const toolNotes = TOOL_CAPABILITY_NOTES[tool] || {}
  return toolNotes[field] || toolNotes['*'] || 'Not returned'
}

export function formatStrategy(value) {
  if (!value) return ''
  return String(value)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export function formatToolList(value) {
  if (!value) return []
  if (Array.isArray(value)) return value.filter(Boolean)
  if (typeof value === 'string') {
    return value
      .split(/[,\s/]+/)
      .map((item) => item.trim())
      .filter(Boolean)
  }
  if (typeof value === 'object') return Object.keys(value).filter(Boolean)
  return []
}
