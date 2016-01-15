import stats
from dataset import Dataset
from dataset import Reader
from prediction_algorithms import NormalPredictor
from prediction_algorithms import BaselineOnly
from prediction_algorithms import KNNBasic
from prediction_algorithms import KNNWithMeans
from prediction_algorithms import KNNBaseline
from prediction_algorithms import CloneBruteforce
from prediction_algorithms import CloneKNNMeanDiff

def evaluate(algo, data):
    for trainset, testset in data.folds:
        algo.train(trainset)
        algo.test(testset)
        stats.compute_stats(algo.preds)


#algo = NormalPredictor(user_based=True)
algo = BaselineOnly(user_based=True)
#algo = KNNBasic(user_based=True)
#algo = KNNWithMeans(user_based=True)
#algo = KNNBaseline(user_based=True, sim_name='pearson_baseline', shrinkage=100)
#algo = CloneBruteforce(user_based=True)
#algo = CloneKNNMeanDiff(user_based=True)

#reader = Reader(line_format='user item rating timestamp', sep='\t')
reader = Reader('ml-100k')

"""
train_file = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u%d.base'
test_file = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]
data = Dataset.load_from_folds(folds_files, reader=reader)
"""

data = Dataset.load('ml-100k')
data.split(n_folds=5)

evaluate(algo, data)



"""
# this is like... cool
algo = KNNBaseline()
data = Dataset.load('ml-100k')
data.split(n_folds=5)
evaluate(algo, data)
"""