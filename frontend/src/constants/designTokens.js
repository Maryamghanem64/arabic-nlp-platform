/**
 * Single source of truth for visual tokens.
 * Keep component styles aligned with these values.
 */

export const TOOL_GROUPS = {
  morphology: ['camel', 'alkhalil'],
  syntax: ['stanza', 'udpipe'],
  segmentation: ['farasa', 'qalsadi'],
}

export const TOOL_COLORS = {
  camel: {
    bg: '#EEF2FF',
    border: '#4F46E5',
    text: '#312E81',
    label: 'CAMeL',
    group: 'morphology',
    groupLabel: 'Morphology',
  },
  alkhalil: {
    bg: '#F5F3FF',
    border: '#7C3AED',
    text: '#5B21B6',
    label: 'AlKhalil',
    group: 'morphology',
    groupLabel: 'Morphology',
  },
  stanza: {
    bg: '#ECFDF5',
    border: '#059669',
    text: '#065F46',
    label: 'Stanza',
    group: 'syntax',
    groupLabel: 'Syntax',
  },
  udpipe: {
    bg: '#ECFDF5',
    border: '#0F766E',
    text: '#134E4A',
    label: 'UDPipe',
    group: 'syntax',
    groupLabel: 'Syntax',
  },
  farasa: {
    bg: '#FFFBEB',
    border: '#D97706',
    text: '#92400E',
    label: 'Farasa',
    group: 'segmentation',
    groupLabel: 'Segmentation',
  },
  qalsadi: {
    bg: '#FFF7ED',
    border: '#EA580C',
    text: '#9A3412',
    label: 'Qalsadi',
    group: 'segmentation',
    groupLabel: 'Segmentation',
  },
}

export const CONFIDENCE_COLORS = {
  high: {
    bg: '#F0FDF4',
    border: '#22C55E',
    text: '#15803D',
    dot: '#22C55E',
    label: 'High',
    threshold: 0.85,
  },
  medium: {
    bg: '#FFFBEB',
    border: '#F59E0B',
    text: '#B45309',
    dot: '#F59E0B',
    label: 'Medium',
    threshold: 0.6,
  },
  low: {
    bg: '#FEF2F2',
    border: '#EF4444',
    text: '#DC2626',
    dot: '#EF4444',
    label: 'Low',
    threshold: 0,
  },
}

export const ACCENT = {
  50: '#EEF2FF',
  100: '#E0E7FF',
  200: '#C7D2FE',
  500: '#6366F1',
  600: '#4F46E5',
  700: '#4338CA',
  800: '#3730A3',
  900: '#312E81',
}

export const SEMANTIC = {
  error: { bg: '#FEF2F2', border: '#FCA5A5', text: '#DC2626' },
  warning: { bg: '#FFFBEB', border: '#FCD34D', text: '#92400E' },
  success: { bg: '#F0FDF4', border: '#86EFAC', text: '#15803D' },
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
  if (score >= CONFIDENCE_COLORS.high.threshold) return 'high'
  if (score >= CONFIDENCE_COLORS.medium.threshold) return 'medium'
  return 'low'
}
