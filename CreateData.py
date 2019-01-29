#from nltk.corpus import genesis as dataset
import os
#https://medium.com/@amarbudhiraja/supervised-language-identification-for-short-and-long-texts-with-code-626f9c78c47c

def create_example_data():
    import nltk
    try:
	os.listdir( nltk.data.find('genesis') )        
	from nltk.corpus import genesis as dataset
	#print(path_to)	
	#from nltk.corpus import genesis as dataset
    except:
        #try:
	#import nltk
        nltk.download('genesis')
	#quit()
	from nltk.corpus import genesis as dataset        
	#except Exception as e:
        #    print(e)
        #    raise EnvironmentError("For Genesis toy data from NLTK you need the Internet access to download it.")
            
    languages = ["finnish", "german", "portuguese","english", "french", "swedish"]

    corpus_words = {"finnish" : list(dataset.words('finnish.txt')) , "german": list(dataset.words('german.txt')),
                    "portuguese": list(dataset.words('portuguese.txt')), "english": list(dataset.words('english-web.txt')),
                    "french": list(dataset.words('french.txt')), "swedish": list(dataset.words('swedish.txt'))}
    return corpus_words
