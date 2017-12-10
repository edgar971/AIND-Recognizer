import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    x_lengths = test_set.get_all_Xlengths()
    sequences_range = range(len(test_set.get_all_sequences()))

    for single_data in sequences_range:
        best_score = float('-inf')
        probability = {}
        x, lengths = x_lengths[single_data]
        
        for word, model in models.items():
            try:
                score = model.score(x, lengths)
                probability[word] = score
                if score > best_score:
                    best_score = score
                    guess_word = word
            except:
                probability[word] = float('-inf')
                
        probabilities.append(probability)
        guesses.append(guess_word)
        
    return probabilities, guesses