import nltk


def preprocess_pipeline(sentences_list: list, remove_empty: bool = True):
    """
    A function to preprocess the sentences in corpus.
    
    :param remove_empty: to remove empty strings
    :param sentences_list: list of list of words in sentences.
    :return: list of list of tokenized words for sentences.
    """
    # Drop Empty Sentences
    if remove_empty:
        sentences = [s for s in sentences_list if len(s) > 0]
    else:
        sentences = sentences_list
    # Empty List to hold Tokenized Sentences
    tokenized = []

    # Iterate through sentences
    for sentence in sentences:
        # create a string out of it
        sentence = ' '.join(sentence)
        # Convert to lowercase
        sentence = sentence.lower()
        # Convert to a list of words
        token = nltk.word_tokenize(sentence)
        # Append to list
        tokenized.append(token)
    return tokenized


def count_the_words(sentences: list):
    """
    A function to count the total number of words in the sentences.
    
    :param sentences: list of lists -> [[tokens in sentence-1], [tokens in sentence-2]].
    :return: word count.
    """
    # Creating a Dictionary of counts
    word_counts = {}

    # Iterating over sentences
    for sentence in sentences:
        # Iterating over Tokens
        for token in sentence:
            # Add count for new word
            if token not in word_counts.keys():
                word_counts[token] = 1
            # Increase count by one
            else:
                word_counts[token] += 1
    return word_counts


def handling_oov(tokenized_sentences: list, count_threshold: int = 6):
    """
    A function to handle out of vocabulary based on a minimum count frequency.
    
    :param tokenized_sentences: all the tokenized sentences.
    :param count_threshold: a threshold to drop any token with frequency lesser than this.
    :return: a closed vocabulary list.
    """
    # Empty list for closed vocabulary
    closed_vocabulary = []

    # Obtain frequency dictionary using previously defined function
    words_count = count_the_words(tokenized_sentences)

    # Iterate over words and counts
    for word, count in words_count.items():

        # Append if it's more(or equal) to the threshold
        if count >= count_threshold:
            closed_vocabulary.append(word)

    return closed_vocabulary


def unk_tokenize(tokenized_sentences: list, vocabulary: list, unknown_token: str = "<unk>"):
    """
    A function to replace every word OOV with an unknown token.
    
    :param tokenized_sentences: list of list of tokens in sentences.
    :param vocabulary: list of all the tokens in vocabulary.
    :param unknown_token: string to replace unknown token with.
    :return: list of tokenized sentences with only tokens in vocabulary.
    """
    # Convert vocabulary list into a set
    vocabulary = set(vocabulary)
    # Create empty list for sentences
    new_tokenized_sentences = []

    # Iterate over sentences
    for sentence in tokenized_sentences:
        # Iterate over sentence and add <unk>
        # if the token is absent from the vocabulary
        new_sentence = []
        for token in sentence:
            if token in vocabulary:
                new_sentence.append(token)
            else:
                new_sentence.append(unknown_token)
        # Append sentece to the new list
        new_tokenized_sentences.append(new_sentence)
    return new_tokenized_sentences


def cleansing(tokenized_sentences, count_threshold):
    """
    A function combining other helper functions.
    
    :param tokenized_sentences: list of list of tokenized sentences.
    :param count_threshold: threshold to keep tokens in vocabulary.
    :return: tokenized train data, vocabulary
    """
    # Get closed Vocabulary
    vocabulary = handling_oov(tokenized_sentences, count_threshold)
    # Updated Training Dataset
    new_train_data = unk_tokenize(tokenized_sentences, vocabulary)

    return new_train_data, vocabulary


def cleansing_test(tokenized_sentences, vocabulary):
    """
        A function combining other helper functions.

        :param tokenized_sentences: list of list of tokenized sentences.
        :param vocabulary: vocabulary to keep words from.
        :return: to drop any word in test that is not in the train vocabulary.
    """
    new_test_data = unk_tokenize(tokenized_sentences, vocabulary)

    return new_test_data
