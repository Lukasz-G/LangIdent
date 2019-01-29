
import numpy, os, codecs, random, sys, copy
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import textwrap

def reading_corpus(path_to_corpus, max_len=None):
    '''
    The corpus directory should be organised 
    as one single folder with subfolders for specific languages,
    each folder for one language containing a number of text files.
    
    Returns a dictionary with different language names as keys 
    and merged text files as values. 
    '''
    languages = {}
    for language in os.listdir(path_to_corpus):
        path_to_language = os.path.join(path_to_corpus,language)
        if os.path.isdir(os.path.join(path_to_language)):
            language_list = []
            for f in os.listdir(path_to_language):
                #if f is a text file????
                t = codecs.open(os.path.join(path_to_language,f), 'r', 'utf-8')
                text_read = []
                for word in t.read().split():
                    text_read.append(word)
                language_list.append(' '.join(text_read))
                t.close()
            if max_len:
                language_list = language_list[:max_len]
            languages[language] = ' '.join(language_list)
            
    return languages

def reading_targets(path_to_folder, max_len=None):
    '''
    The directory with target files should be organised 
    as one single folder with at least one target file with unknown language.
    
    Returns a dictionary with text file contents; 
    names as keys and text files as values. 
    '''
    targets = {}
    for singular_file in os.listdir(path_to_folder):
                #if f is a text file????
                t = codecs.open(os.path.join(path_to_folder,singular_file), 'r', 'utf-8')
                text_read = []
                for word in t.read().split():
                    text_read.append(word)
                targets[singular_file] = ' '.join(text_read)
                if max_len:
                    targets[singular_file] = targets[singular_file][:max_len]
                t.close()
                #print(targets[singular_file].encode('utf-8'))
                #quit()
    return targets

def floatrgb(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try: 
        x = float(mag-cmin)/(cmax-cmin)
    except ZeroDivisionError: 
        x = 0.5 # cmax == cmin
    blue  = min((max((4*(0.75-x), 0.)), 1.))
    red   = min((max((4*(x-0.25), 0.)), 1.))
    green = 0.0#min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return int(red*255), int(green*255), int(blue*255)

def visualise_ppm(ukn, case, lang, ukn_doc_split_tokens, prob_string, path_to_save):
    '''
    A function for visualising the character probability distribution as predicted
    by PPM for an unknown text. Here simplified for LangIdent utility.
    '''
    '''define basic variables needed for further process'''
    characters = set(ukn_doc_split_tokens)
    characters = ' '.join(characters)

    prob_string_values = sorted(list(set(prob_string)))
    max_value = max(prob_string_values)
    min_value = min(prob_string_values)
    
    colours = [floatrgb(p, min_value, max_value) for p in prob_string_values]     
    #characters_ = [characters for x in range(len(colours))]
    text_string = ''.join(ukn_doc_split_tokens)

    text_wrapped = textwrap.wrap(text_string, width=100)

    MAX_W, MAX_H = 1200, 1200

    '''customise the font size to the image size'''
    fontsize = 1  # starting font size
    # portion of image width you want text width to be
    img_fraction_w = 0.50
    img_fraction_h = 0.8
    font = ImageFont.truetype("georgia.ttf", fontsize)
    while font.getsize(text_wrapped[0])[0] < img_fraction_w*MAX_W \
    and font.getsize(text_wrapped[0])[1] * (len(text_wrapped)) < img_fraction_h*MAX_H:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("georgia.ttf", fontsize)
    #optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype("georgia.ttf", fontsize)
    h_font = font.getsize(text_wrapped[0])[1]
    h_colours = int(h_font*2*len(colours))
    #make_color = lambda : (random.randint(50, 255), random.randint(50, 255), random.randint(50,255))
    '''create additional image with all characters'''
    image = Image.new("RGB", (MAX_W, h_colours ), color='white') # scrap image
    draw = ImageDraw.Draw(image)
    xcuts = [draw.textsize(characters[:i+1], font=font)[0] for i in range(len(characters))]
    xcuts = [0]+xcuts
    ycut = draw.textsize(characters, font=font)[1]
    for nb_, colour_characters in enumerate(colours):
        draw.text((0,nb_*ycut*2), characters, colour_characters,font=font)
    ichars_colours = dict()
    for nb_, colour_characters in enumerate(prob_string_values):
        ichars = dict([(characters[i], (xcuts[i+1]-xcuts[i]+1, image.crop((xcuts[i]-1, nb_*ycut*2, xcuts[i+1], nb_*ycut*2+ycut)))) for i in range(len(xcuts)-1)])
        ichars_colours[colour_characters] = ichars
    
    '''create the final image with a target text'''
    image2 = Image.new("RGB", (MAX_W, MAX_H ), color='white') # final image
    fill = " o "
    x = 0
    case_ = case.split('_')
    title = "Text {} with the {} {} for {} language".format(ukn,case_[0], case_[1],lang)
    title_text= textwrap.wrap(title, width=100)
    font2 = ImageFont.truetype("georgia.ttf", fontsize+10)
    title_width = font2.getsize(title_text[0])[0]
    start_for_title = MAX_W/2 - title_width/2
    draw = ImageDraw.Draw(image2)
    draw.text((start_for_title, 10), title_text[0], fill='black',font=font2)
    
    
    pad = ycut
    current_h = MAX_H/10#ycut
    nb_char = 0
    for line in text_wrapped:
        
        current_w = MAX_W/5#10
        for letter in line:

            char_prob_value = prob_string[nb_char]
            w, char_image = ichars_colours[char_prob_value][letter]

         
            image2.paste(char_image, (current_w, current_h))
            current_w += w
        
            nb_char += 1
        current_h = current_h + pad
    
    f_name = '{}.png'.format(ukn+'_'+case)
    path_to_file = os.path.join(path_to_save,f_name)
    image2.save(path_to_file)
    return
   

def mark_ngrams(text, text_as_ngram, ngrams_list, ngram_size):
    '''mark position of frequent ngrams'''
    text_as_place_list = [0.0]*len(text)
    for ngram in ngrams_list:
        indcs = [pos for pos, char in enumerate(text_as_ngram) if char == ngram]
        for i in indcs:
            for r in range(ngram_size):
                text_as_place_list[i+r] = 1.0
    return text_as_place_list

def select_best_worse(matrix):
    '''select the best and the worst predictions'''
    random_file_nb = random.choice(range(matrix.shape[0]))
    random_array = matrix[random_file_nb]
    best = numpy.argmax(random_array)
    worst = numpy.argmin(random_array)
    return (random_file_nb, best), (random_file_nb, worst)

def identify_best_worse(best,worst,lang_names, file_names):
    '''identify names of best and worst language predictions'''
    best_lang = lang_names[best[1]]
    worst_lang = lang_names[worst[1]]
    print(best[0])
    file_name = file_names[best[0]]
    
    return file_name, best_lang, worst_lang


def make_visualisation(matrix_results, languages, 
                       ukn_texts , ukn_names, 
                       ngram_size, ngrams_list,
                       path_to_save):
    '''meta-function for producing text visualisation'''
    best, worst = select_best_worse(matrix_results)
    file_name, best_lang, worst_lang = identify_best_worse(best=best, worst=worst, lang_names=languages, file_names=ukn_names)
    selected_text = ukn_texts[best[0]]
    
    selected_ngram_sublist_best = ngrams_list[best[1]]
    selected_ngram_sublist_worst = ngrams_list[worst[1]]
    #print(selected_ngram_sublist_best)
    #quit()
    text_as_ngram = text_as_ngram_list(text=selected_text, ngram_size=ngram_size)
    
    #print(ngram_size)
    #quit()
    text_as_place_best = mark_ngrams(text=ukn_texts[best[0]], 
                        text_as_ngram=text_as_ngram, 
                        ngrams_list=selected_ngram_sublist_best,
                        ngram_size=ngram_size)
    
    text_as_place_worst = mark_ngrams(text=ukn_texts[best[0]], 
                        text_as_ngram=text_as_ngram, 
                        ngrams_list=selected_ngram_sublist_worst, 
                        ngram_size=ngram_size)
    
    visualise_ppm(ukn=file_name,
                  case = 'best_prediction', 
                  lang = best_lang,
                  ukn_doc_split_tokens=selected_text, 
                  prob_string=text_as_place_best, 
                  path_to_save=path_to_save)
    
    visualise_ppm(ukn=file_name, 
                  case = 'worst_prediction',
                  lang = worst_lang,
                  ukn_doc_split_tokens=selected_text, 
                  prob_string=text_as_place_worst, 
                  path_to_save=path_to_save)
    
def text_as_ngram_list(text, ngram_size):
    '''transform text to a stream of ngrams'''
    ngrams_lang = []
    for i in range(len(text)):
        if i < len(text)-ngram_size: #watch out to not ...
            ngram = text[i:i+ngram_size]
            ngrams_lang.append(ngram)
    return ngrams_lang
def select_ngrams(list_of_languages, ngram_size=1, max_nb_ngrams=100):
    '''choose most frequent ngrams'''
    list_lang_ngrams= []
    for l in list_of_languages:
        ngrams_lang = text_as_ngram_list(text=l, ngram_size=ngram_size)
        cnt = Counter(ngrams_lang)
        mst = cnt.most_common(max_nb_ngrams)
        mst = [x for x,y in mst]
        list_lang_ngrams.append(mst)
        
    return list_lang_ngrams
def check_ngrams(list_of_unknows, list_of_langs):
    '''count ngrams in unknown files'''
    counts_for_all_unknows = []
    for unknow in list_of_unknows:
        counts_for_diff_langs = []
        for lang in list_of_langs:
            nb_counts_per_lang = 0
            for specific_ngram in lang:
                nb_counts = unknow.count(specific_ngram)
                nb_counts_per_lang += nb_counts
            counts_for_diff_langs.append(nb_counts_per_lang)
        counts_for_all_unknows.append(counts_for_diff_langs)
    counts_matrix = numpy.asanyarray(counts_for_all_unknows)
    
    return counts_matrix

def compute_best_results(results,unknown_texts_names,list_labels):
    '''select best scores from the result matrix'''
    bests = []
    for each_unknown in range(results.shape[0]):
        best = numpy.argmax(results[each_unknown])
        unknown_name = unknown_texts_names[each_unknown]
        candidate = list_labels[best]
        bests.append([unknown_name, candidate])
    return bests

def create_training_and_test(list_of_languages, test_set_size=0.2):
    '''create training and test data sets, probably from NTKL Genesis Corpus'''
    test_data_list = []
    training_data_list = []
    nb_frag = 0
    k = 10
    list_frag_names = []
    for l in list_of_languages:
        l_length = len(l)
        test_size = int(test_set_size*l_length)
        random_start = random.choice(range(l_length-test_size))
        test_data = l[random_start:random_start+test_size]
        del l[random_start:random_start+test_size]
        chunk_size = len(test_data)/k
        test_fragments = [test_data[i:i+chunk_size] for i in range(0, test_size, chunk_size)][:-1]
        test_fragments = [' '.join(frag) for frag in test_fragments]
        test_data_list.extend(test_fragments)
        nb_frag_names = ["frag_no_{}".format(x) for x in range(nb_frag,nb_frag+len(test_fragments))]
        nb_frag += k
        list_frag_names.extend(nb_frag_names)
        training_data_list.append(' '.join(l))
    combined = list(zip(test_data_list, list_frag_names))
    random.shuffle(combined)
    test_data_list[:], list_frag_names[:] = zip(*combined)
    
    
    return training_data_list, test_data_list, list_frag_names

def print_results(list_of_bests, short=True, unknown_files=[],path_to_output_dir=None):
    '''printing results into the console'''
    for nb, b in enumerate(list_of_bests):
        language_name = str(b[1]).capitalize()
        if short:
            print("{} is written in {} language".format(b[0],language_name))
        else:
            print("\n File {} with the following text: \n {} \n is written in {} language".format(b[0], unknown_files[nb][:100].encode('utf-8'),language_name))
    
    print("Output data have been saved in your output directory under {}".format(path_to_output_dir))
      
def write_in_results(list_of_bests, path_to_output_dir=None):
    '''writing in results into a specified file'''
    path_to_file_ = os.path.join(path_to_output_dir, 'language identification.txt')
    
    with codecs.open(path_to_file_, mode='w+', encoding='utf-8') as f:
        for b in list_of_bests:
            language_name = str(b[1]).capitalize()
            f.write("{} is written in {} language".format(b[0],language_name))
            f.write('\n')