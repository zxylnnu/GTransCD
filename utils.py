import numpy as np
from sklearn.decomposition import PCA
import random

def GT_To_One_Hot(gt, class_count):

    GT_One_Hot = []
    [height, width]=gt.shape
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot

def get_Samples_GT(seed: int, gt: np.array, class_count: int, train_ratio,val_ratio):

    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []

    train_number_per_class=[]
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32')+\
                                     np.ceil(samplesCount*val_ratio).astype('int32'))
        train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)
    train_rand_idx = np.array(train_rand_idx)
    train_data_index = []
    val_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = list(train_rand_idx[c])
        train_data_index=train_data_index+a[:train_number_per_class[c]]
        val_data_index=val_data_index+a[train_number_per_class[c]:]

    train_data_index = set(train_data_index)
    val_data_index=set(val_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)

    test_data_index = all_data_index - train_data_index - val_data_index

    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)
    val_data_index = list(val_data_index)

    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    return train_samples_gt, test_samples_gt, val_samples_gt