import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import GTransCD
from utils import get_Samples_GT,GT_To_One_Hot
from SegmentMap import SegmentMap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Seed_List = [0,1,2,3,4]
torch.cuda.empty_cache()
OA_ALL = [];AA_ALL = [];KPP_ALL = [];AVG_ALL = [];Train_Time_ALL=[];Test_Time_ALL=[]

data_mat = sio.loadmat('XXX.mat')
data     = data_mat['XXX']
data_mat_be = sio.loadmat(r'XXX1.mat')
data_be     = data_mat_be['XXX1']
data_mat_af = sio.loadmat(r'XXX2.mat')
data_af     = data_mat_af['XXX2']
gt_mat = sio.loadmat(r'XXX_gt.mat')
gt     = gt_mat['XXX_gt']

curr_train_ratio = 0.009
val_ratio = 0.001
class_count = 2
learning_rate = 5e-4
max_epoch = 100
dataset_name = "river"

height, width, bands = data.shape
data    = np.reshape(data, [height * width, bands])
data_be = np.reshape(data_be, [height * width, bands])
data_af = np.reshape(data_af, [height * width, bands])

minMax = preprocessing.StandardScaler()
data    = minMax.fit_transform(data)
data_be = minMax.fit_transform(data_be)
data_af = minMax.fit_transform(data_af)
data    = np.reshape(data, [height, width, bands])
data_be = np.reshape(data_be, [height, width, bands])
data_af = np.reshape(data_af, [height, width, bands])

gt_reshape = np.reshape(gt, [-1])
samplesCount_list = []
for i in range(class_count):
    idx = np.where(gt_reshape == i + 1)[-1]
    samplesCount = len(idx)
    samplesCount_list.append(samplesCount)
print(samplesCount_list)

for curr_seed in Seed_List:

    train_samples_gt, test_samples_gt, val_samples_gt = get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio, val_ratio)

    train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
    test_samples_gt_onehot  = GT_To_One_Hot(test_samples_gt,  class_count)
    val_samples_gt_onehot   = GT_To_One_Hot(val_samples_gt,   class_count)
    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
    test_samples_gt_onehot  = np.reshape(test_samples_gt_onehot,  [-1, class_count]).astype(int)
    val_samples_gt_onehot   = np.reshape(val_samples_gt_onehot,   [-1, class_count]).astype(int)
    Test_GT = np.reshape(test_samples_gt, [height, width])

    train_val_test_gt = [train_samples_gt, val_samples_gt, test_samples_gt]
    for type in range(3):
        gt_reshape = np.reshape(train_val_test_gt[type], [-1])
        samplesCount_list = []
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            samplesCount_list.append(samplesCount)
        print(samplesCount_list)

    train_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    train_samples_gt = np.reshape(train_samples_gt, [height * width])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_count])

    val_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    val_samples_gt = np.reshape(val_samples_gt, [height * width])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_count])

    test_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_count])

    SM = SegmentMap()
    Q_mat, A_mat = SM.getHierarchy()


    Q_mat_gpu = torch.from_numpy(np.array(Q_mat, dtype=np.float32)).to(device)
    A_mat_gpu = torch.from_numpy(np.array(A_mat, dtype=np.float32)).to(device)

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
    train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

    net_input = np.array(data, np.float32)
    net_input_be = np.array(data_be, np.float32)
    net_input_af = np.array(data_af, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)
    net_input_be = torch.from_numpy(net_input_be.astype(np.float32)).to(device)
    net_input_af = torch.from_numpy(net_input_af.astype(np.float32)).to(device)

    net = GTransCD.gtrans(height, width, bands, class_count, Q_mat_gpu, A_mat_gpu).to(device)

    def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
        real_labels = reallabel_onehot

        we = -torch.mul(real_labels, torch.log(predict + 1e-15))
        we = torch.mul(we, reallabel_mask)

        pool_cross_entropy = torch.sum(we)
        return pool_cross_entropy

    zeros = torch.zeros([height * width]).to(device).float()

    def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot):
            with torch.no_grad():
                available_label_idx = (train_samples_gt != 0).float()
                available_label_count = available_label_idx.sum()
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction.cpu() / available_label_count
                return OA

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)

    best_OA = 0
    net.train()

    for i in range(max_epoch + 1):

        optimizer.zero_grad()
        output = net(net_input, net_input_be, net_input_af)
        loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()

        if i % 10 == 0:
            with torch.no_grad():
                net.eval()
                output = net(net_input, net_input_be, net_input_af)
                valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                print("{}\tval OA={}".format(str(i + 1),valOA,))

                if  valOA > best_OA:
                    best_OA = valOA
                    torch.save(net.state_dict(), r"model\best_model.pt")
                    print('save model')

            torch.cuda.empty_cache()
            net.train()

    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(r"model\best_model.pt"))
        net.eval()
        output = net(net_input, net_input_be, net_input_af)
        testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot)

        print("\t===test OA={}===".format(testOA))

    torch.cuda.empty_cache()
    del net
