# -*- coding: utf-8 -*-

"""

Lukasz Gagala
LangIdent for language identification by most frequent character ngrams
Darmstadt assignment

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

...

"""

import argparse, time, logging
from Utils import reading_corpus, reading_targets,\
print_results, write_in_results,\
select_ngrams, check_ngrams, compute_best_results,\
create_training_and_test, make_visualisation


def main():
    
    parser = argparse.ArgumentParser(description='Lukasz Gagala LangIdent for language identification by most frequent ngrams')
    parser.add_argument('-c', type=str,
                        help='Path to language training data')
    parser.add_argument('-t', type=str,
                        help='Path to a folder with target text files')
    parser.add_argument('-o', type=str,
                        help='Path to a file with classification results')
    parser.add_argument('--ngram_size', type=int, default = 3,
                        help='the size of ngrams')
    parser.add_argument('--max_nb_ngrams', type=int, default = '100', 
                        help='the maximal number of ngrams extracted from a language data')
    parser.add_argument('--short', action='store_true',
                        help='Print a short answer. Otherwise a long one.')
    parser.add_argument('--genesis_toy_data', action='store_true',
                        help='Use toy data from NTKL Genesis Corpus. Training data from the path will be ignored')
    parser.add_argument('--visualisation_example', action='store_true',
                        help='Produce a visualisation example with randomly chosen test file: one image with best ngrmas scores, the other one with the worst ones')
    
    
    args = parser.parse_args()
    
    if not args.c and not args.genesis_toy_data:
        raise ValueError("The path to language training data is required. Otherwise use Genesis toy data.")
    if not args.t and not args.genesis_toy_data:
        raise ValueError("The path to a folder with target text files is required. Otherwise use Genesis toy data.")
    if not args.o:
        logging.info(args)
        raise ValueError("The path to a file with classification results is required")
    logging.info(args)
    
    
    """
    Depending on user-specified arguments proceed either with the toy data taken from NLTK corpora
    or read in text data from specified paths.
    """
    if args.genesis_toy_data:
        '''import data from NLTK; otherwise redundant.'''
        from CreateData import create_example_data
        toy_data = create_example_data()
        '''create training and test data'''
        training_data_list, test_data_list, list_frag_names = create_training_and_test(list_of_languages=toy_data.values(), test_set_size=0.3)
        '''select most frequent ngrams'''
        langs_list = select_ngrams(list_of_languages=training_data_list, ngram_size=args.ngram_size, max_nb_ngrams=args.max_nb_ngrams) 
        '''count ngrams'''
        results_matrix = check_ngrams(list_of_unknows=test_data_list, list_of_langs=langs_list)
        '''select best candidates'''
        answers = compute_best_results(results=results_matrix,
                         unknown_texts_names=list_frag_names,
                         list_labels=toy_data.keys())
        '''optionally produce a text visualisation'''
        if args.visualisation_example:
            make_visualisation(matrix_results=results_matrix, languages=toy_data.keys(), 
                               ukn_texts=test_data_list, ukn_names=list_frag_names, 
                               ngram_size=args.ngram_size, ngrams_list=langs_list,
                               path_to_save=args.o)
        '''print and write in results'''
        print_results(list_of_bests=answers, short=args.short, 
                      unknown_files=test_data_list, path_to_output_dir=args.o)
        write_in_results(list_of_bests=answers, path_to_output_dir=args.o)
        
            
        
    else:
        '''create training and test data'''
        training_texts = reading_corpus(args.c)
        targets = reading_targets(args.t)
        '''select most frequent ngrams'''
        langs_list = select_ngrams(list_of_languages=training_texts.values(), ngram_size=args.ngram_size, max_nb_ngrams=100) 
        '''count ngrams'''
        results_matrix = check_ngrams(list_of_unknows=targets.values(), list_of_langs=langs_list)
        '''select best candidates'''
        answers = compute_best_results(results=results_matrix,
                             unknown_texts_names=targets.keys(),
                             list_labels=training_texts.keys())
        '''optionally produce a text visualisation'''
        if args.visualisation_example:
            make_visualisation(matrix_results=results_matrix, languages=training_texts.keys(), 
                               ukn_texts=targets.values(), ukn_names=targets.keys(), 
                               ngram_size=args.ngram_size, ngrams_list=langs_list,
                               path_to_save=args.o)
        '''print and write in results'''
        print_results(list_of_bests=answers, short=args.short, 
                      unknown_files=targets.values(), path_to_output_dir=args.o)
        write_in_results(list_of_bests=answers, path_to_output_dir=args.o)
if __name__ == '__main__':
    main()
