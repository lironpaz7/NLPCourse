import time
import nltk
import utils
import gensim.downloader as api

nltk.download('punkt')
import warnings
from sklearn.svm import SVC
from sklearn.metrics import f1_score

warnings.filterwarnings(action='ignore')

print('[1/6] Reading data...')
X_train, y_train = utils.read_data(path='data/train.tagged')
X_test, y_test = utils.read_data(path='data/dev.tagged')

print('[2/6] Generating context...')
# X_train = utils.generate_context(X_train, constants.TOKEN, method='pn')
# X_test = utils.generate_context(X_test, constants.TOKEN, method='pn')
# s = 0
# for i in X_test:
#     s += len(i)
# print(s, len(y_test))
#


n = 100
print('[3/6] Loading pretrained model...')
model_w2v = api.load(f"glove-twitter-{n}")

print('[4/6] Vectorizing data...')
X_train = utils.vectorize_data(X_train, model_w2v, n=n)
X_test = utils.vectorize_data(X_test, model_w2v, n=n)

print('[5/6] Fitting model...')
start = time.time()
model = SVC()
model.fit(X_train, y_train)
print(f'Fitting took {time.time() - start} seconds')

y_pred = utils.get_predict(X_test, model)
print(f'[6/6] F1 Score: {f1_score(y_test[:-10], y_pred)}')
