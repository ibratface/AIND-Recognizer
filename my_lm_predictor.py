from asl_data import SinglesData
from asl_utils import show_errors

from string import digits
import pickle
import arpa
import itertools

LM_SCALE = 150

lm1 = arpa.loadf("ukn.1.lm")[0]
lm2 = arpa.loadf("ukn.2.lm")[0]
lm3 = arpa.loadf("ukn.3.lm")[0]

exceptional_words = {
    'SAY-P': 'SAY',
    'IX-P': 'IX',
}

with open('probabilities.pickle', 'rb') as file:
    gm = pickle.load(file) # gesture model

with open('test_set.pickle', 'rb') as file:
    test_set = pickle.load(file)

def clean_word(word):
    w = word[:-1] if word[-1].isdigit() else word
    return exceptional_words.get(w, w)

def lm_log_p(sentence):
    # clean_sentence = [ clean_word(word) for word in sentence ]
    try:
        return lm3.log_p(' '.join(sentence))
    except:
        print ('Exception', sentence)

def iterative():
    guesses = []
    for video_num in test_set.sentences_index:
        guess = []
        lm_guess = []
        # iteratively extend the sentence getting the max gesture model prob and language model prob
        for i in test_set.sentences_index[video_num]:
            log_p, word = max([ (gm_log_p + LM_SCALE * lm_log_p(lm_guess + [clean_word(word)]), word) for word, gm_log_p in gm[i].items() ])
            guess.append(word)
            lm_guess.append(clean_word(word))
        guesses.extend(guess)

    show_errors(guesses, test_set)

def build_sentence(first_log_p, first_word, sentences_index):
    sentence = [first_word]
    clean_sentence = [clean_word(first_word)]
    total_log_p = first_log_p
    for i in sentences_index:
        log_p, word = max([ (gm_log_p + LM_SCALE * lm_log_p(clean_sentence + [clean_word(word)]), word) for word, gm_log_p in gm[i].items() ])
        total_log_p += log_p
        sentence.append(word)
        clean_sentence.append(clean_word(word))
    return total_log_p, sentence

def top_iterative():
    guesses = []
    # guess sentence-wise
    for video_num in test_set.sentences_index:
        first = test_set.sentences_index[video_num][0]
        top_first_words = sorted([(gm_log_p + LM_SCALE * lm_log_p([clean_word(word)]), word) for word, gm_log_p in gm[first].items()], reverse=True)[:5]
        top_sentences = [ build_sentence(log_p, w, test_set.sentences_index[video_num][1:]) for log_p, w in top_first_words ]
        _, best = max(top_sentences)
        guesses.extend(best)
    show_errors(guesses, test_set)

def combinations():
    guesses = []
    for video_num in test_set.sentences_index:
        # get the top 3 candidate words from our gesture model (for each word in the sentence)
        top_words = [sorted(gm[i].items(), key=lambda item: item[1], reverse=True)[:3] for i in test_set.sentences_index[video_num]]
        sentences = []
        # for each permutation of words, calculate the log_p
        for permutations in itertools.product(*top_words):
            sentence = [ word for word, _ in permutations ]
            clean_sentence = [ clean_word(word) for word, _ in permutations ]
            log_p = sum([ gm_log_p for _, gm_log_p in permutations ]) + LM_SCALE * lm_log_p(clean_sentence)
            sentences.append((log_p, sentence))
        _, best = max(sentences)
        print (best)
        guesses.extend(best)
    show_errors(guesses, test_set)

def main():
    top_iterative()

if __name__ == "__main__":
    main()
