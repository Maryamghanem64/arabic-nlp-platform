# Evaluation Methodology

## Purpose

This platform performs agreement-based, capability-aware evaluation of Arabic NLP analyzer outputs. It does not evaluate against a human-annotated gold standard, and it does not claim that analyzer agreement equals linguistic correctness.

The evaluation layer answers a narrower research question: when multiple tools are capable of producing a given linguistic feature, how similar are their outputs on the submitted text?

## Core Principle

Evaluation is scoped by capability. A tool is included in a metric only when all of the following are true:

- The tool supports the evaluated feature.
- The tool is available for the current run.
- The tool produced comparable values after normalization and alignment.
- The tool is not lazy, loading, unavailable, timed out, missing resources, excluded, or otherwise degraded for that feature.

Unsupported or unavailable tools are excluded from the denominator. They are not counted as wrong outputs.

## Evaluated Features

The evaluation endpoint reports:

- POS agreement
- Lemma match, including exact and normalized match fields
- Root agreement
- Segmentation coverage
- Active tools
- Excluded tools
- Capability contributors
- Metric contributors
- Alignment metadata
- Methodology notes and degraded-tool notes

These metrics are agreement indicators, not accuracy, precision, recall, or F1 against a gold reference. Backward-compatible fields named `pos_precision`, `pos_recall`, and `pos_f1` are agreement-derived proxy fields and should not be described as supervised accuracy metrics.

## Capability Contributors

The backend uses capability sets for each metric:

- POS: CAMeL, Stanza, UDPipe, SinaTools, AlKhalil
- Lemma: CAMeL, Stanza, Qalsadi, AlKhalil, UDPipe, SinaTools
- Root: CAMeL, AlKhalil, SinaTools
- Segmentation: Farasa, CAMeL, AlKhalil, SinaTools
- Dependency: Stanza, UDPipe
- Contextual: AraBERT, reported separately when available

MADAMIRA is excluded from current evaluation because required licensed resources are not available in the defense configuration. AraBERT is listed only as contextual evidence and is excluded from direct morphology, lemma, root, POS, segmentation, and dependency metrics.

## Excluded Statuses

The evaluation service excludes tools with statuses such as:

- `timeout`
- `unavailable`
- `future_work`
- `lazy`
- `disabled`
- `lazy_not_loaded`
- `loading`
- `missing_resources`
- `excluded`
- `skipped_low_memory`

Farasa receives explicit degraded handling. If Farasa times out or returns an unavailable/degraded status, it remains visible to the UI but is removed from segmentation scoring for that run.

## Normalization and Alignment

Before evaluation, raw analyzer outputs are normalized into a common token structure. Normalization extracts or preserves:

- Surface form
- Lemma
- Root
- POS or UPOS
- Segmentation
- Dependency evidence
- Morphological features
- Confidence metadata when present

The alignment layer selects a dynamic available base, preferring Farasa when usable and falling back to other available token streams when necessary. This avoids making the entire evaluation structurally dependent on Farasa.

## POS Agreement

POS agreement is computed only from POS-capable tools that are active and comparable in the current run. Tool-specific labels are normalized before comparison. AlKhalil POS evidence is canonicalized against contextual votes to reduce false conflicts caused by label-format differences.

The POS score is the average majority-agreement ratio across aligned tokens with at least two POS contributors.

## Lemma Match

Lemma match is agreement-based. Lemmas are normalized by removing comparison-irrelevant variation such as diacritics and certain orthographic differences before voting.

The endpoint exposes exact and normalized lemma fields for frontend compatibility, but the current service computes them as agreement-derived indicators rather than gold-standard matches.

## Root Agreement

Root agreement compares normalized root evidence from root-capable tools. Diacritics and separators are normalized before comparison. Functional-word roots should be interpreted carefully because roots are less meaningful for many particles, prepositions, and conjunctions.

## Segmentation Coverage

Segmentation coverage reports whether aligned tokens have usable segmentation evidence from segmentation-capable contributors. It is a coverage/completeness signal, not a correctness score.

Farasa is the strongest segmentation contributor, but segmentation disagreement often reflects different clitic-splitting conventions rather than analyzer failure.

## Tool-Specific Methodology Notes

- CAMeL: primary morphology, lemma, root, and POS contributor.
- Farasa: primarily a segmentation contributor; timeout/degraded behavior is excluded from scoring.
- Stanza: POS, lemma, and UD syntax contributor; tokenization and multi-word-token behavior can differ from other analyzers.
- Qalsadi: lemma-focused partial contributor.
- AlKhalil: morphology/root contributor; POS labels require canonical normalization.
- UDPipe: UD POS and dependency contributor.
- SinaTools: local lexical-resource contributor; included only when loaded and comparable.
- AraBERT: contextual support only; excluded from direct morphology metrics.
- MADAMIRA: excluded because required licensed resources are missing.

## Interpretation Rules

When reading the UI or reports:

- Agreement means similarity between analyzer outputs.
- Coverage means available evidence, not correctness.
- Conflict counts show disagreement among comparable outputs.
- Confidence values summarize available evidence; they are not gold-standard probabilities.
- Excluded/lazy/unavailable tools should be visible but should not reduce metric scores.
