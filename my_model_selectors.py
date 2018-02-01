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
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bestmodel = None
        bestscore = float("inf")
        num_features = self.X.shape[1]
        #log of data points
        logN = np.log(len((self.lengths)))

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Train HMM model
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                # Log Likelihood on trained HMM
                logL = hmm_model.score(self.X, self.lengths)
                # complexity penalty, The BIC applies a larger penalty when N > e^2 = 7.4.
                p = num_states + (num_states * (num_states - 1)) + (num_states * num_features * 2)
                # Calculate Bayesian Information Criteria (BIC)
                # BIC = −2 log L + p log N
                BIC_score = -2 * logL + p * logN
            except:
                continue

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

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Num of words
        M = len((self.words).keys())

        bestmodel = None
        bestscore = float('-inf')

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Train HMM
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
            except:
                logL = float("-inf")

            log_sum = 0
            for word in self.hwords.keys():
                ix_word, word_lengths = self.hwords[word]

            try:
                log_sum += hmm_model.score(ix_word, word_lengths)

            except:
                log_sum += 0

            # Update discriminant Score
            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(for all except i))
            DIC_Score = logL - (1 / (M - 1)) * (log_sum - (0 if logL == float("-inf") else logL))

            # update best model
            if DIC_Score > bestscore:
                bestscore = DIC_Score
                bestmodel = hmm_model

        return bestmodel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Q: implement model selection using CV
        bestscore = float('-inf')
        bestmodel = None

        # Check for folds sufficiency
        if len(self.sequences) < 2:
            return None

        #fold if n>2
        kf = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            logsum, count = 0, 0
            # Iterate on sequences split from Kfold
            for cv_train_ix, cv_test_ix in kf.split(self.sequences):
                X_train, lengthX_train = combine_sequences(cv_train_ix, self.sequences)
                X_test, lengthX_test = combine_sequences(cv_test_ix, self.sequences)
                try:
                    # Train HMM
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengthX_train)
                    logL = hmm_model.score(X_test, lengthX_test)
                    count += 1
                except:
                    logL = 0
                logsum += logL

            # calculate final cv score
            cv_score = logsum / (1 if count == 0 else count)

            # select best model
            if cv_score > bestscore:
                bestscore = cv_score
                bestmodel = hmm_model

        return bestmodel