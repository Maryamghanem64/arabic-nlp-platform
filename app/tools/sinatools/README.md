SinaTools demo resources
========================

Place the demo lemma dictionary here:

    app/tools/sinatools/lemma.pickle

The upstream SinaTools package expects `lemmas_dic.pickle` plus optional n-gram
pickles under `%APPDATA%/sinatools`. The platform adapter keeps the demo
self-contained by accepting `lemma.pickle` in this folder.
