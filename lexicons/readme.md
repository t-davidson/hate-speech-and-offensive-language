## Hate Speech Lexicons

This directory contains two lexicons that can be used to identify hate speech.

The file `hatebase_dict.csv` contains the original lexicon from [Hatebase.org](https://www.hatebase.org) that we used to sample tweets. While this lexicon can achieve high recall it is associated with a high rate of false positives as it contains many words that are generally not used in an offensive or hateful manner (e.g. yellow, oreo, bird).

The file `refined_ngram_dict.csv` contains a refined lexicon of n-grams. To get this lexicon we took the set of n-grams of length 1-4 that were contained in our labelled data and for each n-gram calculated the proportion of tweets containing it that were considered as hate speech by the human coders. We then manually went through the lexicon to remove irrelevant terms.
