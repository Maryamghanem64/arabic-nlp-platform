export function canonicalToken(raw) {
  if (!raw || typeof raw !== 'object') return {}

  const analysis = firstAnalysis(raw)
  const merged = {
    ...analysis,
    ...raw,
  }

  merged.surface = firstDefined(raw.surface, raw.word, analysis.surface, merged.surface)
  merged.lemma = firstDefined(raw.lemma, analysis.lemma, merged.lemma)
  merged.root = firstDefined(raw.root, analysis.root, merged.root)
  merged.pos = firstDefined(raw.pos, raw.upos, analysis.pos, analysis.upos, merged.pos)
  merged.gloss = firstDefined(raw.gloss, analysis.gloss, merged.gloss)
  merged.segmentation = firstDefined(raw.segmentation, raw.segments, raw.parts, analysis.segmentation, analysis.segments, analysis.parts, merged.segmentation)
  merged.dependency = raw.dependency || analysis.dependency || merged.dependency || null
  merged.features = raw.features || analysis.features || merged.features || null
  merged.original_surface = firstDefined(raw.original_surface, analysis.original_surface, merged.original_surface)
  merged.normalized = Boolean(raw.normalized ?? analysis.normalized ?? merged.normalized)
  merged.note = firstDefined(raw.note, analysis.note, merged.note)
  merged.analyses = Array.isArray(raw.analyses) ? raw.analyses : Array.isArray(analysis.analyses) ? analysis.analyses : []

  return merged
}

export function firstAnalysis(raw) {
  if (!raw || typeof raw !== 'object') return {}
  if (Array.isArray(raw.analyses) && raw.analyses.length > 0 && raw.analyses[0] && typeof raw.analyses[0] === 'object') {
    return raw.analyses[0]
  }
  if (raw.final && typeof raw.final === 'object') {
    return raw.final
  }
  return {}
}

export function firstDefined(...values) {
  for (const value of values) {
    if (value === null || value === undefined) continue
    if (Array.isArray(value)) {
      if (value.length > 0) return value
      continue
    }
    if (typeof value === 'string') {
      if (value.trim()) return value
      continue
    }
    return value
  }
  return ''
}
