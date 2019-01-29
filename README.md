# LangIdent
LangIdent for simple language identification by most frequent character ngrams

There is also an additional mini-corpus included for Darmstadt Assignment - with text samples for Dutch, Frisian, Low German, English and German.

Dependencies:
- Python 2.7 or 3.6 (the Anaconda Python dystribution recommended)
- argparse, time, os, sys, 
- json, math, copy, math, logging, 
- numpy, codecs, random, 
 -textwrap, PIL 
(all of them included in the aforementioned distribution of Anaconda)

Usage from the command line:
>>> python LangIdent.py -c PATH-TO-TRAINING-DATA -t PATH-TO-TEST-DATA -o PATH-TO-RESULTS
    [-ngram_size INT-FOR-THE-NGRAM-SIZE] 
    [--max_nb_ngrams MAX-NUMBER-OF-NGRAMS] 
   

Example of usage:
>>>  python LangIdent.py -c /path/to/your/data -t path/to/test/data -o /where/you/want/to/save
[--ngram_size 3] [--max_nb_ngrams 100] [--short] [--genesis_toy_data] [--visualisation_example]

References:

http://practicalcryptography.com/miscellaneous/machine-learning/tutorial-automatic-language-identification-ngram-b/
http://cloudmark.github.io/Language-Detection/
