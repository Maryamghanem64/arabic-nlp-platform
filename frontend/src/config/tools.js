import { TOOL_COLORS } from '@/constants/designTokens'

function tool(key, meta) {
  const visual = TOOL_COLORS[key] || {}
  return {
    key,
    label: visual.label || key,
    color: visual.border || '#64748B',
    group: visual.group || 'unknown',
    groupLabel: visual.groupLabel || 'Unknown',
    ...meta,
  }
}

export const TOOL_CONFIG = {
  camel: tool('camel', {
    type: 'Hybrid morphological analysis',
    license: 'MIT',
    features: ['Lemma', 'Root', 'POS', 'Morphological features', 'Gloss'],
    paper: 'Obeid et al. 2020, ACL',
    provides: ['lemma', 'root', 'pos', 'gender', 'number', 'tense', 'gloss'],
    researchRole: 'Primary lexical and morphological evidence',
  }),
  alkhalil: tool('alkhalil', {
    type: 'Rule-based morphology',
    license: 'Free for research',
    features: ['Lemma', 'Root', 'Morphological analysis'],
    paper: 'Boudchiche et al. 2017',
    provides: ['lemma', 'root', 'pos'],
    researchRole: 'Rule-based morphological support',
  }),
  sinatools: tool('sinatools', {
    type: 'Lexical and morphological NLP',
    license: 'MIT',
    features: ['Lemmatization', 'POS tagging', 'Lexical evidence'],
    paper: 'Jarrar et al., Birzeit University',
    provides: ['lemma', 'pos', 'root'],
    researchRole: 'Optional lexical-morphology evidence when local resources are loaded',
  }),
  madamira: tool('madamira', {
    type: 'Statistical + rule-based morphology',
    license: 'LDC / research',
    features: ['Morphology', 'Diacritization', 'POS'],
    paper: 'Pasha et al. 2014, LREC',
    provides: ['lemma', 'pos', 'gender', 'number'],
    researchRole: 'Optional morphological analyzer; excluded when not configured',
  }),
  stanza: tool('stanza', {
    type: 'Neural UD pipeline',
    license: 'Apache 2.0',
    features: ['POS', 'Lemma', 'Dependency', 'Morphological features'],
    paper: 'Qi et al. 2020, ACL',
    provides: ['lemma', 'pos', 'case', 'definite', 'dependency'],
    researchRole: 'UD-oriented syntactic evidence',
  }),
  udpipe: tool('udpipe', {
    type: 'UD parsing pipeline',
    license: 'MPL 2.0',
    features: ['POS', 'Dependency parsing', 'Lemma'],
    paper: 'Straka 2018, CoNLL',
    provides: ['lemma', 'pos', 'dependency'],
    researchRole: 'Independent UD syntax support',
  }),
  farasa: tool('farasa', {
    type: 'Statistical segmentation',
    license: 'Research use',
    features: ['Segmentation', 'Clitic boundaries'],
    paper: 'Abdelali et al. 2016, NAACL',
    provides: ['segmentation'],
    researchRole: 'Segmentation anchor',
  }),
  qalsadi: tool('qalsadi', {
    type: 'Rule-based lexical morphology',
    license: 'LGPL',
    features: ['Lemma', 'Stem'],
    paper: 'Zerrouki 2017',
    provides: ['lemma', 'stem'],
    researchRole: 'Rule-based lexical support',
  }),
  arabert: tool('arabert', {
    type: 'Contextual Transformer model',
    license: 'Apache 2.0',
    features: ['Contextual embeddings', 'Semantic representation'],
    paper: 'Antoun et al. 2020, LREC',
    provides: [],
    researchRole: 'Contextual representation; not a direct morphology table competitor',
  }),
}

export const TOOL_KEYS = Object.keys(TOOL_CONFIG)

export const FEATURE_ELIGIBILITY = {
  segmentation: ['farasa'],
  lemma: ['camel', 'alkhalil', 'sinatools', 'stanza', 'udpipe', 'qalsadi'],
  root: ['camel', 'alkhalil', 'sinatools'],
  pos: ['camel', 'sinatools', 'stanza', 'udpipe', 'alkhalil'],
  dependency: ['stanza', 'udpipe'],
}

export function toolOrder(keys = TOOL_KEYS) {
  return keys.filter((key) => TOOL_CONFIG[key])
}

export function toolMeta(key) {
  return TOOL_CONFIG[key] || {
    key,
    label: key || 'Unknown tool',
    color: '#64748B',
    group: 'unknown',
    groupLabel: 'Unknown',
    type: 'Optional analyzer',
    provides: [],
    researchRole: 'No frontend metadata registered',
  }
}

export function eligibleTools(feature, keys = TOOL_KEYS) {
  const eligible = new Set(FEATURE_ELIGIBILITY[feature] || [])
  return keys.filter((key) => eligible.has(key))
}
