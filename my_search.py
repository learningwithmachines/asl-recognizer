import arpa
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ngramguess(object):
    '''
    ngrams -class interface
    '''

    def __init__(self, lm_model: arpa.models.simple.ARPAModelSimple, sentence: dict, num_words_to_test=5, hyper=0.9):
        '''

        :param lm_model: the lm model, specified by the arpa module and loaded *.lm file
        :param sentence: dict, probabilities of words in sentence sequence from test_setd.
        :param num_words_to_test: int, size of ngram, length of words to compare against
                                the ones in LM sentence sequence
        :param hyper: int, scores scaling factor
        '''
        self.lm_model = lm_model
        self.sentence = sentence
        self.ngramsize = num_words_to_test
        self.hyper = hyper

    def score(self):
        #preallocating.. list for guess words and HMM probabilities
        guesses = [[] for x in range(self.ngramsize)]
        probs = [[] for x in range(self.ngramsize)]
        bestguess = ''
        bestscore = float('-inf')
        lmscore, hmmscore = [], []
        #functions
        gets = lambda x: self.sentence[XWORD][x]
        mmaxscaler = MinMaxScaler()
        
        probs_np_arr = np.array(probs)
        #normalizing our best guesses probabilities with MinMaxScaler
        scaled_probs = mmaxscaler.fit_transform(probs_np_arr)

        # find best guesses at each word in sentence, and append word and probs,
        # and pop best guess from the lookup dictionary for next best guess.
        for ID, XWORD in enumerate(self.sentence):
            for numguess in range(self.ngramsize):
                guesses[numguess].append(max(self.sentence[XWORD], key=gets ))
                probs[numguess].append(self.sentence[XWORD][guesses[numguess][ID]])
                self.sentence[XWORD].pop(guesses[numguess][ID], None)

        #from the best guesses, make ngrams (size: numtrials ** ngramsize)
        sentences, word_score = [], []
        #cleanup for copies, e.g 'GIVE1' and 'GIVE2' are copies of 'GIVE'
        for word_ in range(len(guesses)):
            for numguess in range(len(guesses[word_])):
                if '0' <= guesses[word_][numguess][-1] and guesses[word_][numguess][-1] <= '9':
                    guesses[word_][numguess] = guesses[word_][numguess][:-1]                      #trim

        # making ngrams
        for trial in range(np.power(len(guesses), len(guesses[0]))):
            numguess = trial
            sentences.append("")
            word_score.append([])
            for k in range(0, len(guesses[0])):
                # build all sequences iteratively
                sentences[trial] += guesses[int(numguess % self.ngramsize)][k] + " "
                word_score[trial].append(scaled_probs[int(numguess % self.ngramsize)][k])
                numguess /= self.ngramsize
        
        # default for best guess
        bestguess = sentences[0]
        # calculate scores (lmmscore, hmmscore) for every possible sentence patterns in set.
        for ID, sentence_ in enumerate(sentences):
            trywords = sentence_.split(' ')
            lmscore.append(0)
            lmscore[ID] += self.lm_model.log_p(trywords[0] + " " + trywords[1])
            for word_ in range(2, len(trywords) - 1):
                lmscore[ID] += np.log10(self.lm_model.p(trywords[word_ - 2] + " " + trywords[word_ - 1] + " " + trywords[word_]))
            
            lmscore[ID] += np.log10(self.lm_model.p(trywords[len(trywords) - 2] + " " + trywords[len(trywords) - 1] + " </s>"))
            lmscore[ID] += np.log10(self.lm_model.p(trywords[len(trywords) - 1] + " </s>"))
            hmmscore.append(0)
            for word_ in range(0, len(trywords) - 1):
                hmmscore[ID] += word_score[ID][word_]

        #Rescale Scores and fudge to avoid true divide errors.
        lmscore -= min(lmscore)
        lmscore /= (max(lmscore) + float('1e-5'))
        hmmscore -= min(hmmscore)
        hmmscore /= (max(hmmscore) + float('1e-5'))

        # apply scaling and find the best scored guess.
        for ID, sentence_ in enumerate(sentences):
            lmscore[ID] += self.hyper * hmmscore[ID]
            if lmscore[ID] > bestscore:
                bestguess = sentence_
                bestscore = lmscore[ID]

        return bestguess