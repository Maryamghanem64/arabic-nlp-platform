export const TOOL_CONFIG = {
  camel: {
    label: 'CAMeL Tools',
    color: '#2E5FA3',
    type: 'Hybrid Rule + Neural',
    license: 'MIT',
    features: ['Lemma', 'Root', 'POS', 'Morphology', 'Gloss'],
    paper: 'Obeid et al. 2020, ACL',
    provides: ['lemma', 'root', 'pos', 'gender', 'number', 'tense', 'gloss'],
  },
  farasa: {
    label: 'Farasa',
    color: '#6C3483',
    type: 'Statistical (SVM-rank)',
    license: 'Free (research)',
    features: ['Segmentation', 'Diacritization'],
    paper: 'Abdelali et al. 2016, NAACL',
    provides: ['segmentation'],
  },
  stanza: {
    label: 'Stanza',
    color: '#1E8449',
    type: 'Neural (BiLSTM)',
    license: 'Apache 2.0',
    features: ['POS', 'Lemma', 'Dependency', 'Case'],
    paper: 'Qi et al. 2020, ACL',
    provides: ['lemma', 'pos', 'case', 'definite', 'dependency'],
  },
  qalsadi: {
    label: 'Qalsadi',
    color: '#D35400',
    type: 'Rule-based morphological',
    license: 'LGPL',
    features: ['Lemma', 'Stem'],
    paper: 'Zerrouki 2017',
    provides: ['lemma', 'stem'],
  },
  arabert: {
    label: 'AraBERT',
    color: '#7D3C98',
    type: 'Neural (BERT)',
    license: 'Apache 2.0',
    features: ['POS', 'NER', 'Contextual embeddings'],
    paper: 'Antoun et al. 2020, LREC',
    provides: ['lemma', 'pos'],
  },
  alkhalil: {
    label: 'AlKhalil',
    color: '#1A5276',
    type: 'Rule-based',
    license: 'Free (research)',
    features: ['Root extraction', 'Full morphology'],
    paper: 'Boudchiche et al. 2017',
    provides: ['lemma', 'root', 'pos'],
  },
  udpipe: {
    label: 'UDPipe 2',
    color: '#117A65',
    type: 'Neural (transition-based)',
    license: 'MPL 2.0',
    features: ['POS', 'Dependency parsing', 'Lemma'],
    paper: 'Straka 2018, CoNLL',
    provides: ['lemma', 'pos', 'dependency'],
  },
  madamira: {
    label: 'MADAMIRA',
    color: '#784212',
    type: 'Statistical + Rule-based',
    license: 'LDC (research)',
    features: ['Full morphology', 'Diacritization', 'POS'],
    paper: 'Pasha et al. 2014, LREC',
    provides: ['lemma', 'pos', 'gender', 'number'],
  },
  sinatools: {
    label: 'SinaTools / Alma',
    color: '#B7950B',
    type: 'Frequency + BERT',
    license: 'MIT',
    features: ['Lemmatization', 'POS tagging', 'NER', 'Word Sense Disambiguation'],
    paper: 'Jarrar et al. 2022, Birzeit University',
    provides: ['lemma', 'pos'],
  },
}

export const TOOL_KEYS = Object.keys(TOOL_CONFIG)

export function toolOrder(keys = TOOL_KEYS) {
  return keys.filter((key) => TOOL_CONFIG[key])
}
