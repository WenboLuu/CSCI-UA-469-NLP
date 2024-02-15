## Homework 3: Implement a Viterbi HMM POS tagger
by Wenbo Lu - Net id: wl2707

Spring 2023

## Package dependancies
    - python3
    - numpy
    - defaultdict (from collections)

## How did I handle OOV:
I adopted a general rule-based treatment for **both** in-vocab words and OOV words.
  1. For in-vocab words, the base emission probability is obtained from the table.
  2. For OOV words, the base emission probability is set to 1e-7.
   
The base emission probability is then multiplied by a list of scaling factors, depending on the word's **morphology**:
(e.g. suffix, prefix, capitalization, etc.). The scaling factors are manually tuned.

   
## How to run:
Call the `HMM.py` directly.

## Performance
A final **95.67%** accuracy on the test set submitted to gradescope.