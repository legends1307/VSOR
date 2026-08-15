import pickle
from pprint import pprint
mat_file = "./data/RVSOD/train"
import os
mat_file = os.path.join(mat_file,"ranking saliency masks", "mat", "actioncliptest00001","actioncliptest00001_196.mat")
import scipy.io as sio
data = sio.loadmat(mat_file)['img']
with open("temp_file.txt",'w') as f:
    for l in data:
        for c in l:
            f.write(str(c))
        f.write("\n")