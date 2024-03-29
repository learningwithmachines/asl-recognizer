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
    # Q: implement the recognizer
    # return probabilities, guesses

    for y, _ in test_set.get_all_Xlengths().items():
        X, Xlength = test_set.get_item_Xlengths(y)

        scoresdict = {}
        bestword = str("")
        bestscore = float("-inf")

        # Repeat for all words and associated HMM models
        for word, model in models.items():
            trialscore = float("-inf")
            try:
                # update score 0-gram
                trialscore = model.score(X, Xlength)
            except:
                pass

            # update bestscore
            if trialscore >= bestscore:
                bestscore = trialscore
                bestword = word

            scoresdict[word] = trialscore

        probabilities.append(scoresdict)
        guesses.append(bestword)
    return probabilities, guesses