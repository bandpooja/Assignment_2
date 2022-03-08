import concurrent
import json
import nltk
from nltk.corpus import brown
import pytrec_eval
from tqdm import tqdm

from utils.preprocess import cleansing, cleansing_test, preprocess_pipeline
from utils.ngram_model import NGramModel
from utils.helper import load_test


if __name__ == "__main__":
    nltk.download('brown')

    # region Model Fitting
    new_list = brown.sents()
    min_freq = 6
    tokenized_sent = preprocess_pipeline(new_list)
    final_train, vocabulary = cleansing(tokenized_sent, min_freq)

    ns = [1, 2, 3, 5]  #, 10]
    model_loc = 'models'

    for n in ns:
        model = NGramModel(n=n, model_loc=model_loc, vocabulary=vocabulary)
        model.fit(final_train)
        model.save_model()
    # endregion

    # region validation
    # previous_tokens = ["the", "jury"]
    #
    # for n in ns:
    #     model = NGramModel(n=n, model_loc=model_loc, vocabulary=vocabulary)
    #     model.load_model()
    #     print(f"{n}-gram model prediction: {model.get_suggestions(previous_tokens)}")
    # endregion

    # region testing
    # Making prediction on all the birdbeck data Just checking success at k for all the predictions
    test_df = load_test(file_loc='data/APPLING1DAT.643')
    tokenized_sent = preprocess_pipeline(test_df['previous-tokens'].values.tolist(), remove_empty=False)
    final_test = cleansing_test(tokenized_sent, vocabulary)
    test_df['final-test'] = final_test

    queries = [{} for _ in ns]
    results_eval = [{} for _ in ns]

    for idx, n in enumerate(ns):
        model = NGramModel(n=n, model_loc=model_loc, vocabulary=vocabulary)
        model.load_model()

        argument_list = final_test
        suggestions = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(model.get_suggestions, argument_list):
                suggestions.append(result)

        query = queries[idx]
        result_eval = results_eval[idx]
        for fill_in_word, test_previous_tokens, suggestion in tqdm(zip(test_df['fill-in-word'], final_test,
                                                                        suggestions), total=len(final_test),
                                                                   desc=f'Predictions-{n}-gram-model'):
            query[f"{' '.join(test_previous_tokens)} *"] = {fill_in_word: 1}
            result_eval[f"{' '.join(test_previous_tokens)} *"] = {}
            print(test_previous_tokens)
            print(fill_in_word)
            print(suggestion)
            for word in [w[0] for w in suggestion[1]]:
                result_eval[f"{' '.join(test_previous_tokens)} *"][word] = 1

            for word in [w[0] for w in suggestion[5]]:
                if word not in result_eval[f"{' '.join(test_previous_tokens)} *"].keys():
                    result_eval[f"{' '.join(test_previous_tokens)} *"][word] = 1 / 5

            for word in [w[0] for w in suggestion[10]]:
                if word not in result_eval[f"{' '.join(test_previous_tokens)} *"].keys():
                    result_eval[f"{' '.join(test_previous_tokens)} *"][word] = 1 / 10

        print('#' * 20)
        print(f'Stats of {n}-gram model')
        print('#' * 20)
        evaluator = pytrec_eval.RelevanceEvaluator(query, {'success'})

        print(json.dumps(evaluator.evaluate(result_eval), indent=1))
        eval = evaluator.evaluate(result_eval)

        for measure in sorted(list(eval[list(eval.keys())[0]].keys())):
            print(measure, 'average:',
                  pytrec_eval.compute_aggregated_measure(
                      measure, [query_measures[measure] for query_measures in eval.values()])
                  )
        print('#' * 20)
    # endregion
