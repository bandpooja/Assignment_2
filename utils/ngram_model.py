import concurrent

from joblib import Memory
memory = Memory("./cache", verbose=0)
import operator
import os
import pickle

from utils.helper import take


def count_n_grams(data, n, start_token="<s>", end_token="<e>"):
    # Empty dict for n-grams
    n_grams = {}

    # Iterate over all sentences in the dataset
    for sentence in data:
        # Append n start tokens and a single end token to the sentence
        sentence = [start_token] * n + sentence + [end_token]
        # Convert the sentence into a tuple
        sentence = tuple(sentence)

        # Temp var to store length from start of n-gram to end
        m = len(sentence) if n == 1 else len(sentence) - 1

        # Iterate over this length
        for i in range(m):
            # Get the n-gram
            n_gram = sentence[i:i + n]

            # Add the count of n-gram as value to our dictionary
            # IF n-gram is already present
            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            # Add n-gram count
            else:
                n_grams[n_gram] = 1
    return n_grams


def prob_for_single_word(word, previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary_size, k=1.0):
    # Convert the previous_n_gram into a tuple

    # if we have lesser available tokens we will pad <s> at front to make the lengths similar
    if len(previous_n_gram) != len(list(nminus1_gram_counts.keys())[0]):
        previous_n_gram = ['<s>'] * (len(list(nminus1_gram_counts.keys())[0])-len(previous_n_gram)) +\
                          list(previous_n_gram)

    previous_n_gram = tuple(previous_n_gram)
    # Calculating the count, if exists from our freq dictionary otherwise zero
    previous_n_gram_count = nminus1_gram_counts[previous_n_gram] if previous_n_gram in nminus1_gram_counts else 0

    # The Denominator
    denom = previous_n_gram_count + k * vocabulary_size
    # previous n-gram plus the current word as a tuple
    n_gram = previous_n_gram + (word,)

    # Calculating the n count, if exists from our freq dictionary otherwise zero
    nplus1_gram_count = n_gram_counts[n_gram] if n_gram in n_gram_counts else 0
    # Numerator
    num = nplus1_gram_count + k

    # Final Fraction
    prob = num / denom
    return prob


def helper_prob_for_single_word(args_):
    return prob_for_single_word(args_[0], args_[1], args_[2], args_[3], args_[4], k=args_[5])


def probs(previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary, k=1.0, parallelize=False):
    # Convert to Tuple
    previous_n_gram = tuple(previous_n_gram)

    # Add end and unknown tokens to the vocabulary
    vocabulary = vocabulary + ["<e>", "<unk>"]
    # Calculate the size of the vocabulary
    vocabulary_size = len(vocabulary)

    # Empty dict for probabilites
    probabilities = {}

    # Iterate over words --> this is the function that needs to be parallelized across cores
    if not parallelize:
        for word in vocabulary:
            # Calculate probability
            probability = prob_for_single_word(word, previous_n_gram,
                                               nminus1_gram_counts, n_gram_counts,
                                               vocabulary_size, k=k)
            # Create mapping: word -> probability
            probabilities[word] = probability
    else:
        results = []
        args_ = [(word, previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary_size, k) for word in vocabulary]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results.append(executor.map(helper_prob_for_single_word, args_))

        for res, word in zip(results, vocabulary):
            probabilities[word] = res

        print(probabilities)
    return probabilities


def top_n_selection(word_probability_tuple: list, top_n: int = 1):
    # assumes that the token_distance list of tuples is sorted with distance
    d_nth = word_probability_tuple[top_n-1][1]
    ix = top_n
    ix_top_n = top_n
    for i in range(ix, len(word_probability_tuple)):
        if word_probability_tuple[i][1] != d_nth:
            break
        else:
            ix_top_n = i+1

    top_n_words = word_probability_tuple[:ix_top_n]
    return top_n_words


def auto_complete(previous_tokens, nminus1_gram_counts, n_gram_counts, vocabulary, k=1.0):
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # most recent 'n-1' words
    '''
        For Example: n-1 because for bi gram model only the last word is given importance.
        i.e. n = 2 then only n-1 which is 1 is actually given importance.
    '''
    previous_n_gram = previous_tokens[-(n-1):]
    # Calculate probabilty for all words
    probabilities = probs(previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary, k=k)

    # sort the dictionary based on values and then return words with probability in top-n
    sorted_pf = [(k, v) for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)]

    # getting top - n words given the history

    # # to assign the words with same probability same rank
    # top_1 = top_n_selection(sorted_pf, top_n=1)
    # top_5 = top_n_selection(sorted_pf, top_n=5)
    # top_10 = top_n_selection(sorted_pf, top_n=10)

    # to assign the words with same probability same rank
    top_1 = [sorted_pf[0]]
    top_5 = sorted_pf[:5]
    top_10 = sorted_pf[:10]

    return {1: top_1, 5: top_5, 10: top_10}


# caching the function output for quick reruns
@memory.cache
def unigram_auto_complete(n_gram_counts, k=0.0):
    # length of previous words
    n = 1

    # Calculate probability for all words
    probabilities = {}
    total_counts = sum(n_gram_counts.values())
    for w, f in n_gram_counts.items():
        probabilities[w] = (f + k)/(total_counts + k)

    # sort the dictionary based on values and then return words with probability in top-n
    # we just need a word not a tuple it
    sorted_pf = [(k[0], v) for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)]
    # getting top - n words given the history

    # # to assign the words with same probability same rank
    # top_1 = top_n_selection(sorted_pf, top_n=1)
    # top_5 = top_n_selection(sorted_pf, top_n=5)
    # top_10 = top_n_selection(sorted_pf, top_n=10)

    # to assign the words with same probability same rank
    top_1 = [sorted_pf[0]]
    top_5 = sorted_pf[:5]
    top_10 = sorted_pf[:10]
    return {1: top_1, 5: top_5, 10: top_10}


class NGramModel:
    def __init__(self, n: int, model_loc: str, vocabulary):
        self.n_gram_counts = {}
        self.n = n
        self.n_gram_counts['n'] = self.n
        self.model_loc = model_loc
        self.vocabulary = vocabulary

    def fit(self, data):
        if self.n > 1:
            n_counts = count_n_grams(data, self.n)
            n_minus_one_counts = count_n_grams(data, self.n-1)
            self.n_gram_counts['n-1_counts'] = n_minus_one_counts
            self.n_gram_counts['n_counts'] = n_counts
        else:
            # unigram model only need to save one probability dictionary
            n_counts = count_n_grams(data, self.n)
            self.n_gram_counts['n_counts'] = n_counts

    def save_model(self):
        os.makedirs(self.model_loc, exist_ok=True)
        with open(os.path.join(self.model_loc, f'{self.n}-gram-counts.pickle'), 'wb') as handle:
            pickle.dump(self.n_gram_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(os.path.join(self.model_loc, f'{self.n}-gram-counts.pickle'), 'rb') as handle:
            self.n_gram_counts = pickle.load(handle)

    def get_suggestions(self, previous_tokens: list, k: float = 1.0):
        if self.n > 1:
            suggestion = auto_complete(previous_tokens, self.n_gram_counts['n-1_counts'],
                                       self.n_gram_counts['n_counts'], self.vocabulary, k=k)
        else:
            # no need to smoothen the uni-gram
            suggestion = unigram_auto_complete(self.n_gram_counts['n_counts'])
        return suggestion
