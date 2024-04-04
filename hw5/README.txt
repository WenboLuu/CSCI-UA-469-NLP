NYU NLP Homework 5: Feature selection for Maxent Group tagger
    by Wenbo Lu

Features selected:
    - word
    - word stem
    - the position of the word in the sentence, normalized by the length of the sentence
    - POS tag
    - PREFIX    
    - SUFFIX
    - whether capitalized
    - previous BIO tag
    - previous word
    - previous POS tagger
    - previous previous word
    - previous previous POS tag
    - next word
    - next POS tag
    - next next word
    - next next POS tag
    - predicted BIO tag using the baseline tagger (noun phrase chunking using regexp and treebank tagger)
    - previous predicted BIO tag
    - next predicted BIO tag

Results:
    Baseline model (noun phrase chunking using regexp and treebank tagger):
        F1: 76.11%
    Maxent model:
        F1: 91.16% without custom prefix and suffix features
        F1: 91.15% with custom prefix and suffix features
        F1: 91.09% with baseline results included

Score on the development set:
    31676 out of 32853 tags correct
    accuracy: 96.42
    8378 groups in key
    8573 groups in response
    7720 correct groups
    precision: 90.05
    recall:    92.15
    F1:        91.09
    rounded to: 91

