import numpy as np
from sklearn.model_selection import KFold


input_data = np.array([1,2,3,4,5,6,7,8,9])
output_data = np.array([10,11,12,13,14,15,16,17,18,19])


cross_fold_object = KFold(n_splits=5, random_state=None, shuffle=False)
for i, (train_index, test_index) in enumerate(cross_fold_object.split(input_data)):

    print("Train Index", train_index)
    print("Test Index", test_index)
