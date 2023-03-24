import pickle
from tqdm import tqdm
from hw1.preprocessing import read_test
from inference import memm_viterbi


def run_test(test_path, pre_trained_weights, feature2id, predictions_path):
    """
    Running the evaluation process with the given weights and features and produces a prediction file
    :param test_path: path to the test file
    :param pre_trained_weights: weights
    :param feature2id: Feature2id object
    :param predictions_path: path to store the model predictions
    """
    test = read_test(test_path, tagged=False)
    output_file = open(predictions_path, "a+")
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        preds = memm_viterbi(sentence, pre_trained_weights, feature2id)
        sentence = sentence[2:]
        for i in range(len(preds)):
            if i > 0:
                output_file.write(" ")
            pred, word = preds[i], sentence[i]
            output_file.write(f"{word}_{pred}")
        output_file.write("\n")
    output_file.close()


def eval_test(weights, features, test=1):
    """
    Interface function to run the 'run_test' function
    :param weights: weights
    :param features: Feature2id object
    :param test: Integer that indicates which file to run whether 1 or 2.
    """
    print(f'Running evaluation on comp{test}.words...')
    run_test(f'./data/comp{test}.words', weights, features, f'comp_m{test}.wtag')
    print('Finished evaluating...')
    print(f'Saving predictions into comp_m{test}_311280283_313535379.wtag...')
    print('Finished!')


for i in range(1, 3):
    data = pickle.load(open(f'weights{i}.pkl', 'rb'))  # (weights, feature) tuple
    weights, features = data
    eval_test(weights[0], features, test=i)
