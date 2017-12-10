import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        components_range = range(self.min_n_components, self.max_n_components + 1)
        best_bic = float('inf')
        best_model = None
        
        for n in components_range:
            try:
                model = self.base_model(n) 
                score = model.score(self.X, self.lengths)
                number_of_params = n * (n - 1) + (2 * mode.n_features * n) - 1
                bic = -2 * score + number_of_params * np.log(self.X.shape[0])
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except:
                pass

        if best_model != None:
            return best_model
        else: 
            return self.base_model(self.n_constant)
            



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        components_range = range(self.min_n_components, self.max_n_components + 1)
        best_dic = float('-inf')
        best_model = None
        
        for n in components_range:
            try:
                model = self.base_model(n)
                log_l = []
                for word in self.words.keys():
                    if word == self.this_word:
                        continue
                    else:
                        log_l.append(model.score(word.X, word.lengths))
                
                dic = model.score(self.X, self.lengths) - np.mean(log_l)
                if dic > best_dic:
                    best_dic = dic
                    best_model = model
            except:
                pass
        
        if best_model:
            return best_model
        
        return self.base_model(self.n_constant)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        components_range = range(self.min_n_components, self.max_n_components + 1)
        best_CV = float('-inf')
        best_model = None
        
        for n in components_range:
            try:
                if(len(self.sequences) < 3):
                  best_model = self.base_model(n)
                  break

                method = KFold(n_splits=3)
                log_l_list = []
                model = self.base_model(n)
                
                for train_idx, test_idx in method.split(self.sequences):
                    train_x, train_lengths = combine_sequences(train_idx, self.sequences)
                    test_x, test_lengths = combine_sequences(test_idx, self.sequences)
                    model = GaussianHMM(n_components=n, covariance_type="diag",
                                                n_iter=1000, random_state=self.random_state,
                                                verbose=False).fit(train_x,train_lengths)
                    score = model.score(test_x, test_lengths)
                    log_l_list.append(score)
                
                average = np.average(log_l_list)
                if average > best_CV:
                    best_CV = average
                    best_model = model
            except:
                pass
        
        if best_model:
            return best_model
        
        return self.base_model(self.n_constant)
