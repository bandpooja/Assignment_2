# first line: 154
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
