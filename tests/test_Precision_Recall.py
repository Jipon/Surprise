

from surprise import SVD
from surprise import Dataset
from surprise import dataset
import os

# First train an SVD algorithm on the movielens dataset.
#data = Dataset.load_builtin('ml-100k')
path1="your custom trainingset"
path2="your custom testset"


file_path = os.path.expanduser(path1)
reader=dataset.Reader(line_format='user item rating', sep=' ')
#data =Dataset.load_from_file('G:\\08.01\\data\\train8\\train.txt',reader)
data =Dataset.load_from_file(file_path ,reader)
trainset = data.build_full_trainset()
algo = SVD()


algo.train(trainset)



#get recommend correctcount ---@k

#top-k

k=5
correctcount = trainset.build_hit_testdata(algo,path2,k)
#print(correctcount)
#get testcount
testcount=len(trainset.read_file(path2))
#get recommend totalcount
totalcount=trainset.n_users*k


print("---------------------")

print("Precision:")
if totalcount!=0:
    print(correctcount/totalcount)
print("Recall:")
if testcount!=0:
    print(correctcount/testcount)
print("--------------------")
