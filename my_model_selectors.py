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

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorBIC, self).__init__(all_word_sequences,
                                          all_word_Xlengths,
                                          this_word,
                                          n_constant=n_constant,
                                          min_n_components=min_n_components,
                                          max_n_components=max_n_components,
                                          random_state=random_state,
                                          verbose=verbose)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestmodel = None
        bestscore = float("+inf")
        alpha = 1.0
        penalty = lambda x: (x[0] - 1) + (x[1] - 1) + x[2] + x[3]
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model, modelparams, logL = None, None, None
            try:
                # Train HMM model
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                modelparams = [hmm_model.startprob_.size,hmm_model.transmat_.size,
                                hmm_model.means_.size,hmm_model.covars_.diagonal().size]
                # Log Likelihood on trained HMM
                logL = hmm_model.score(self.X, self.lengths)
            except:
                pass
            # num_features = self.X.shape[1]
            # complexity penalty, The BIC applies a larger penalty when N > e^2 = 7.4.
            # The number of parameter can also be calculated by our hmmlearn model
            # P = (model.startprob_.size - 1) + (model.transmat_.size - 1) + model.means_.size + model.covars_.diagonal().size
            p = penalty(modelparams)
            logN = np.log(len((self.X)))
            # Calculate Bayesian Information Criteria (BIC)
            # BIC = −2 log L + alpha * p log N
            BIC_score = -2 * logL + alpha * p * logN
            # Select best model
            if bestscore > BIC_score:
                bestmodel = hmm_model
                bestscore = BIC_score

        return bestmodel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorDIC, self).__init__(all_word_sequences,
                                          all_word_Xlengths,
                                          this_word,
                                          n_constant=n_constant,
                                          min_n_components=min_n_components,
                                          max_n_components=max_n_components,
                                          random_state=random_state,
                                          verbose=verbose)

    def is_copy(self, word):
        w = word
        thisword = self.this_word
        is_copy = False

        if "0" <= w[-1] and "9" >= w[-1]:
            if thisword == w[:-1]:
                is_copy = True
            if thisword == w[:-1]:
                is_copy = True

        if "0" <= thisword[-1] and "9" >= thisword[-1]:
            if w == thisword[:-1]:
                is_copy = True
            if w[:-1] == thisword[:-1]:
                is_copy = True

        return is_copy

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestmodel = None
        bestscore = float('-inf')
        alpha = 1.0
        # Num of words
        M = len((self.words).keys())
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # try Training HMM
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
            except:
                # ignore this model if model is None
                logL = float("-inf")
                hmm_model = None
            log_sum = 0
            # check if model predicts bad scores for other words.
            for word in self.hwords.keys():
                ix_word, word_lengths = self.hwords[word]
                if word == self.this_word:
                    continue
                elif self.is_copy(word):
                    try:
                        # scaling == 0.2 for copywords
                        log_sum += 0.2 * hmm_model.score(ix_word, word_lengths)
                        M -= 1
                    except:
                        log_sum += 0
                        M -= 1
                else:
                    try:
                        log_sum += hmm_model.score(ix_word, word_lengths)
                    except:
                        log_sum += 0
                        M -= 1
            # Update discriminant Score
            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(for all except i))
            DIC_Score = logL - (alpha / max(1, (M - 1))) * (log_sum - logL)
            # update best model
            if DIC_Score > bestscore:
                bestscore = DIC_Score
                bestmodel = hmm_model

        return bestmodel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorCV, self).__init__(all_word_sequences,
                                         all_word_Xlengths,
                                         this_word,
                                         n_constant=n_constant,
                                         min_n_components=min_n_components,
                                         max_n_components=max_n_components,
                                         random_state=random_state,
                                         verbose=verbose)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestmodel = None
        bestscore = float('-inf')

        for n in range(self.min_n_components, self.max_n_components + 1):
            Logscores = []
            if len(self.sequences) == 2:
                n_splits = 2
            else:
                n_splits = min(3, len(self.sequences))
            try:
                KF = KFold(n_splits=n_splits)
                hmm_model = None
                for cv_train_idx, cv_test_idx in KF.split(self.sequences):
                    try:
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        hmm_model = GaussianHMM(n_components=n, covariance_type='diag', n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        Logscores.append(hmm_model.score(X_test, lengths_test))
                    except:
                        pass
                if len(Logscores) > 0:
                    avg_score = sum(Logscores) / float(len(Logscores))
                    if avg_score > bestscore:
                        bestscore = avg_score
                        bestmodel = hmm_model
            except:
                pass

        return bestmodel
