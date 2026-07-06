/**
 * Single source of truth for visual tokens.
 * Keep component styles aligned with main.css.
 *
 * Research UI rule:
 * - tool colors identify analyzer families
 * - semantic colors identify status: agreement, warning, conflict, unavailable
 */

export const TOOL_GROUPS = {
  morphology: ['camel', 'alkhalil', 'madamira', 'sinatools'],
  syntax: ['stanza', 'udpipe'],
  segmentation: ['farasa'],
  lexicalSupport: ['qalsadi'],
  contextual: ['arabert'],
}

export const TOOL_COLORS = {
  camel: {
    bg: '#EEF4F9',
    border: '#315C8C',
    text: '#244A73',
    label: 'CAMeL',
    group: 'morphology',
    groupLabel: 'Morphology',
    provides: ['lemma', 'root', 'pos', 'morphology'],
  },
  alkhalil: {
    bg: '#F5F3FF',
    border: '#7A6F8F',
    text: '#554A6B',
    label: 'AlKhalil',
    group: 'morphology',
    groupLabel: 'Morphology',
    provides: ['lemma', 'root', 'morphology'],
  },
  madamira: {
    bg: '#F4F1FA',
    border: '#6F5E8F',
    text: '#4F4169',
    label: 'MADAMIRA',
    group: 'morphology',
    groupLabel: 'Morphology',
    provides: ['lemma', 'pos', 'morphology', 'disambiguation'],
  },
  sinatools: {
    bg: '#F1F5F9',
    border: '#5B6F8A',
    text: '#334155',
    label: 'SinaTools',
    group: 'morphology',
    groupLabel: 'Morphology',
    provides: ['lemma', 'pos', 'lexical evidence'],
  },
  stanza: {
    bg: '#F0F7F4',
    border: '#5F7F78',
    text: '#365F56',
    label: 'Stanza',
    group: 'syntax',
    groupLabel: 'Syntax',
    provides: ['pos', 'lemma', 'dependency', 'morphology'],
  },
  udpipe: {
    bg: '#EEF7F7',
    border: '#4F7C80',
    text: '#285D61',
    label: 'UDPipe',
    group: 'syntax',
    groupLabel: 'Syntax',
    provides: ['pos', 'lemma', 'dependency'],
  },
  farasa: {
    bg: '#FBF7EE',
    border: '#A47C48',
    text: '#75572F',
    label: 'Farasa',
    group: 'segmentation',
    groupLabel: 'Segmentation',
    provides: ['segmentation', 'clitic boundaries'],
  },
  qalsadi: {
    bg: '#F8F4EF',
    border: '#8A7357',
    text: '#634F3B',
    label: 'Qalsadi',
    group: 'lexicalSupport',
    groupLabel: 'Lexical support',
    provides: ['lemma support', 'rule-based support'],
  },
  arabert: {
    bg: '#F1F5F9',
    border: '#64748B',
    text: '#334155',
    label: 'AraBERT',
    group: 'contextual',
    groupLabel: 'Contextual',
    provides: ['contextual embeddings', 'semantic support'],
  },
}

export const CONFIDENCE_COLORS = {
  high: {
    bg: '#F0F7F4',
    border: '#5F7F78',
    text: '#365F56',
    dot: '#5F7F78',
    label: 'High',
    threshold: 0.85,
  },
  medium: {
    bg: '#FBF7EE',
    border: '#A47C48',
    text: '#75572F',
    dot: '#A47C48',
    label: 'Medium',
    threshold: 0.6,
  },
  low: {
    bg: '#FBF2F2',
    border: '#A85C5C',
    text: '#7E3F3F',
    dot: '#A85C5C',
    label: 'Low',
    threshold: 0,
  },
}

export const ACCENT = {
  50: '#EEF4F9',
  100: '#DDEAF3',
  200: '#C9D8E6',
  500: '#4F7DAA',
  600: '#315C8C',
  700: '#244A73',
  800: '#1E3A5F',
  900: '#122A42',
}

export const SEMANTIC = {
  success: {
    bg: '#F0F7F4',
    border: '#5F7F78',
    text: '#365F56',
    label: 'Agreement',
  },
  warning: {
    bg: '#FBF7EE',
    border: '#A47C48',
    text: '#75572F',
    label: 'Partial / warning',
  },
  error: {
    bg: '#FBF2F2',
    border: '#A85C5C',
    text: '#7E3F3F',
    label: 'Conflict / error',
  },
  unavailable: {
    bg: '#F3F5F7',
    border: '#CBD5E1',
    text: '#667085',
    label: 'Unavailable / N/A',
  },
}

export const RADIUS = {
  control: '8px',
  card: '12px',
  pill: '20px',
  chip: '6px',
}

export const FONT_WEIGHT = {
  regular: 400,
  medium: 500,
  semibold: 600,
}

export const FONT_SIZE = {
  label: '11px',
  caption: '12px',
  small: '13px',
  body: '15px',
  lead: '18px',
  h3: '18px',
  h2: '22px',
  h1: '28px',
  arabic: '1.4rem',
}

export function getConfidenceLevel(score) {
  const numeric = Number(score)
  if (!Number.isFinite(numeric)) return 'low'
  if (numeric >= CONFIDENCE_COLORS.high.threshold) return 'high'
  if (numeric >= CONFIDENCE_COLORS.medium.threshold) return 'medium'
  return 'low'
}

export function getToolColor(toolKey) {
  return TOOL_COLORS[String(toolKey || '').toLowerCase()] || {
    bg: '#F3F5F7',
    border: '#CBD5E1',
    text: '#667085',
    label: toolKey || 'Unknown',
    group: 'unknown',
    groupLabel: 'Unknown',
    provides: [],
  }
}
