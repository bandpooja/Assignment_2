from itertools import islice
import pandas as pd


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def load_test(file_loc):
    with open(file_loc) as f:
        lines = f.readlines()

    test_lines = []
    correct_words = []
    incorrect_words = []

    for line_ in lines:
        if '$' != line_[0]:
            line__ = line_.replace('\n', '')
            word_sets = [w for w in line__.split(' ') if len(w) > 0]
            incorrect_words.append(word_sets[0])
            correct_words.append(word_sets[1])
            test_lines.append(' '.join(word_sets[2:]))

    fill_in_char = '*'
    previous_tokens = []
    for line_ in test_lines:
        words = line_.split(' ')
        fill_in_loc = [i for i, w in enumerate(words) if w == fill_in_char][0]
        previous_tokens.append(words[:fill_in_loc])

    test_df = pd.DataFrame()
    test_df['fill-in-word'] = correct_words
    test_df['previous-tokens'] = previous_tokens
    test_df['test-seq'] = test_lines
    return test_df



