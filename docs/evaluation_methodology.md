# Evaluation Methodology

## Purpose

This platform compares the outputs of multiple Arabic NLP analyzers. It does not evaluate against a human-annotated gold standard, and it does not claim that tool agreement equals linguistic correctness.

Agreement values on this platform describe similarity between analyzer outputs only.

## What Is Evaluated

The evaluation layer reports:

- POS agreement
- Lemma agreement
- Exact lemma match
- Normalized lemma match
- Segmentation coverage
- Root agreement
- Processing time
- Active and excluded tools

The evaluation output is produced from the backend responses returned by the analyzers and the comparison/alignment services.

## Input Source

The input is user-submitted Arabic text.

- In the frontend, the user types or selects a sentence.
- The frontend sends that sentence to the backend `/compare` and `/evaluate` endpoints.
- The backend runs the available analyzers on the same text.

## Token Normalization

Before comparison, each analyzer output is normalized into a common token schema.

Normalization includes:

- extracting the token surface form
- extracting lemma, root, POS, gloss, segmentation, dependency, features, and confidence when present
- converting tool-specific POS labels to comparable POS labels
- preserving missing values as missing rather than inventing replacements

The compare flow then aligns tokens by surface string and segmentation-aware span matching.

## Agreement Calculation

Agreement is a similarity score based on the outputs of the active analyzers.

It is not accuracy.
It is not a gold-standard score.
It is not a correctness label.

### Weighted POS Agreement

POS agreement is computed from the aligned tokens using tool weights:

- CAMeL: `0.35`
- Stanza: `0.35`
- UDPipe: `0.15`
- Qalsadi: `0.10`
- AlKhalil: `0.05`

For each aligned token:

1. Collect the available normalized POS values from the tools.
2. Ignore missing values and the generic `X` label.
3. Sum the weights for each distinct POS value.
4. Divide the highest supported weight by the total supported weight.

The reported POS agreement is the weighted majority ratio, averaged across aligned tokens that have POS evidence.

### Lemma Agreement

Lemma agreement is computed twice:

- exact lemma agreement
- normalized lemma agreement

The computation uses the same weighted majority logic as POS agreement.

For normalized lemma agreement, each lemma is passed through the platform's lemma normalization helper before voting.

### Root Agreement

Root agreement is computed on aligned tokens by comparing the available root values after stripping Arabic diacritics.

The backend counts a token as root-agreeing when the available root values match the reference root chosen for that token.

### Segmentation Comparison

Segmentation is compared using Farasa as the preferred segmentation source when it is available.

The backend compares the aligned Farasa segmentation with the aligned base-token segmentation by joining the segments into strings and checking whether they match.

This is a comparison of analyzer outputs, not a validation against an external gold corpus.

## Coverage

Coverage describes how much aligned token evidence is present for the current sentence.

The evaluation endpoint reports segmentation coverage as the proportion of aligned tokens that have usable segmentation evidence in the base alignment.

Coverage is a completeness indicator, not a correctness score.

## Processing Time

Processing time is reported as runtime measured around the frontend request and, where available, as per-tool runtime in the backend payload.

The frontend measures elapsed request time with `performance.now()`.
The backend also attaches runtime fields such as `runtime_ms` or `elapsed` when a tool response includes them.

## Penn Arabic Treebank Note

This platform does not directly evaluate against the Penn Arabic Treebank because it is a licensed corpus.
Some integrated NLP tools were originally developed or evaluated using ATB in their respective publications.

That historical context does not mean this platform uses ATB as its own evaluation reference.

## Interpretation Rules

When reading the UI:

- Agreement means similarity between tool outputs
- Coverage means available evidence, not correctness
- Conflict counts show disagreement across tools
- Confidence values are backend-derived evidence summaries

The Evaluation and Compare pages must not present majority agreement as correctness.
