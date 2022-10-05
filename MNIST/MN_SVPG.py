# -*- coding:utf-8 -*-



# ~~~~~~~~~~~~~~~~~~~~LIBRARIES~~~~~~~~~~~~~~~~~~~~~
# -----basic-------------
import sys
import os
import time
import datetime
import numpy as np
import random
import pickle
import math
# import zipfile
# -----torch-------------
import torch
import torch.nn as nn
import torch.nn.functional as func
# -----image-------------
import cv2
import skimage
import argparse


parser = argparse.ArgumentParser(description='Description: PLACEHOLDER')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise', type=float, default=0.02)
parser.add_argument('--hid_num', type=int, default=20)
parser.add_argument('--hid_dim', type=int, default=10)
parser.add_argument('--sparsity', type=float, default=0.1)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


# ~~~~~~~~~~~~~~~~~~~~CONFIGURATION~~~~~~~~~~~~~~~~~
os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda

R_SEED = 2022 + args.rep
torch.manual_seed(R_SEED)
torch.cuda.manual_seed_all(R_SEED)
np.random.seed(R_SEED)
random.seed(R_SEED)
torch.backends.cudnn.deterministic = True
ENV_SEED = 0

TIME_STAMP = 'MNIST_WTA_rep%d' % (args.rep)
# TIME_STAMP = 'MNIST_WTA'
# TIME_STAMP = 'MNIST_WTA_trial_%4.3f_%3.2f_%2.1f_%d_%d_r%2d' % (args.lr, args.noise, args.entropy, args.hid_num, args.hid_dim, args.rep)


TRAIN_STEP_NUM = 50000
BATCH_SIZE = 100
DIM_STATE = 784
DIM_ACTION = 10
DEVICE = torch.device('cuda:0')
File = open('./log/log_' + TIME_STAMP + '.txt', 'w')


# ~~~~~~~~~~~~~~~~~~~~FUNCTIONS~~~~~~~~~~~~~~~~~~~~~
def index2onehot(index_data, dim_num):
    onehot_data = np.zeros([index_data.shape[0], dim_num], dtype=np.float32)
    onehot_data[np.arange(index_data.shape[0]), index_data] = 1
    return onehot_data


def log_text(rec_text, progress=None):
    if progress is not None:
        print(rec_text, '  ', end='')
        visualize_test_score(progress)
    else:
        print(rec_text)
    File.write(rec_text + '\n')
    File.flush()


def visualize_test_score(test_sc):
    ratio = test_sc * 1.0
    print_number = int(ratio * 50)
    for _ in range(print_number):
        print('|', end='')
    print('')



# ~~~~~~~~~~~~~~~~~~~~LOAD DATA~~~~~~~~~~~~~~~~~~~~~
DATA_MNIST_FILE = open('./MNIST_DATA/mnist.pkl', 'rb')
DATA_MNIST = pickle.load(DATA_MNIST_FILE)
D_TR_IMG = DATA_MNIST['training_images'] / 255
D_TR_LAB = DATA_MNIST['training_labels']
D_TE_IMG = DATA_MNIST['test_images'] / 255
D_TE_LAB = DATA_MNIST['test_labels']
# -----reshape-----------
D_TR_IMG_RS = D_TR_IMG.reshape([-1, 28, 28])
D_TE_IMG_RS = D_TE_IMG.reshape([-1, 28, 28])
# -----onehot------------
D_TE_LAB_onehot = index2onehot(D_TE_LAB, DIM_ACTION)
D_TR_LAB_onehot = index2onehot(D_TR_LAB, DIM_ACTION)
# -----CUDA--------------
D_TR_IMG = torch.from_numpy(D_TR_IMG).to(DEVICE).float()
D_TE_IMG = torch.from_numpy(D_TE_IMG).to(DEVICE).float()
D_TR_LAB_onehot = torch.from_numpy(D_TR_LAB_onehot).to(DEVICE).float()
D_TE_LAB_onehot = torch.from_numpy(D_TE_LAB_onehot).to(DEVICE).float()


# ~~~~~~~~~~~~~~~~~~~~NETWORK~~~~~~~~~~~~~~~~~~~~~~~
class NetWTA:
    def __init__(self):
        self.dev = DEVICE
        # -----Shape params------
        self.dim_state = 784
        self.dim_action = 10
        self.num_hidden = args.hid_num
        self.dim_hidden = args.hid_dim
        self.num_action = 1
        # -----Shape sizes-------
        self.dim_h = self.num_hidden * self.dim_hidden
        self.dim_a = self.num_action * self.dim_action
        self.dim_s = self. dim_state
        self.dim_ha = self.dim_h + self.dim_a
        self.dim_has = self.dim_h + self.dim_a + self.dim_s
        # -----W and b-----------
        self.weight = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
        self.index_1x, self.index_1y = torch.triu_indices(self.dim_ha, self.dim_ha)
        self.index_2x, self.index_2y = torch.tril_indices(self.dim_ha, self.dim_ha)
        self.weight[self.index_2x, self.index_2y] = self.weight[self.index_1x, self.index_1y]
        self.bias = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
        # -----Mask--------------
        self.mask_weight = torch.ones([self.dim_has, self.dim_ha], device=self.dev)
        for index in range(0, self.dim_h, self.dim_hidden):
            self.mask_weight[index:(index + self.dim_hidden), index:(index + self.dim_hidden)] = 0
        for index_x in range(0, self.dim_h, self.dim_hidden):       # Circuit-level sparse connection
            for index_y in range(0, self.dim_h, self.dim_hidden):
                if random.random() < args.sparsity:
                    self.mask_weight[index_x:(index_x + self.dim_hidden), index_y:(index_y + self.dim_hidden)] = 0
                    self.mask_weight[index_y:(index_y + self.dim_hidden), index_x:(index_x + self.dim_hidden)] = 0
        self.mask_weight[self.dim_h:self.dim_ha, self.dim_h:self.dim_ha] = 0
        self.weight = self.weight * self.mask_weight

    def hid_act_softmax(self, q_hid_act):
        input_batch_size = q_hid_act.shape[0]
        q_hid = q_hid_act[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        q_hid_reshape_soft = func.softmax(q_hid, dim=2)
        q_hid_soft = q_hid_reshape_soft.reshape([input_batch_size, self.dim_h])
        q_act = q_hid_act[:, self.dim_h:self.dim_ha].reshape([input_batch_size, self.num_action, self.dim_action])
        q_act_reshape_soft = func.softmax(q_act, dim=2)
        q_act_soft = q_act_reshape_soft.reshape([input_batch_size, self.dim_a])
        q_hid_act_soft = torch.cat([q_hid_soft, q_act_soft], dim=1)
        return q_hid_act_soft

    def forward(self, x):
        input_batch_size = x.shape[0]
        # Get Probability
        q_ha = self.hid_act_softmax(torch.rand([input_batch_size, self.dim_ha], dtype=torch.float32, device=self.dev))
        q_s = x.clone()
        for iter_i in range(50):
            q_has = torch.cat([q_ha, q_s], dim=1)
            q_has = torch.clamp(q_has + args.noise * torch.randn(q_has.size(), device=self.dev), 0, 1)
            q_has[:, 0:self.dim_ha] = q_has[:, 0:self.dim_ha] * 2
            temp_q_ha = torch.mm(q_has, self.weight) + self.bias.expand([input_batch_size, self.dim_ha])
            target_q_ha = self.hid_act_softmax(temp_q_ha)
            if torch.mean(torch.abs(q_ha - target_q_ha)).item() < 0.005:
                break
            q_ha = target_q_ha
        if iter_i > 40:
            print(iter_i)
        q_has = torch.cat([q_ha, q_s], dim=1)
        # Get Sample
        v_hid_act = torch.zeros_like(q_ha, device=self.dev)
        q_hid = q_ha[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        hid_dist = torch.distributions.OneHotCategorical(q_hid)
        hid_sample = hid_dist.sample()
        v_hid_act[:, 0:self.dim_h] = hid_sample.reshape([input_batch_size, self.dim_h])
        q_act = q_ha[:, self.dim_h:self.dim_ha].reshape([input_batch_size, self.num_action, self.dim_action])
        act_dist = torch.distributions.OneHotCategorical(q_act)
        action_dist_entropy = act_dist.entropy()
        act_sample = act_dist.sample()
        v_hid_act[:, self.dim_h:self.dim_ha] = act_sample.reshape([input_batch_size, self.dim_a])
        return q_has, q_ha, v_hid_act, act_sample, action_dist_entropy

    def save_model(self):
        torch.save(self.weight, './log/' + TIME_STAMP + '_w_best.pt')
        torch.save(self.bias, './log/' + TIME_STAMP + '_b_best.pt')
        torch.save(self.mask_weight, './log/' + TIME_STAMP + '_m_best.pt')



model_WTA = NetWTA()

test_accuracy = 0
max_accuracy = 0
max_accuracy_epi_i = 0
train_accuracy_list = []

for train_step_i in range(TRAIN_STEP_NUM):
    sample_list = random.sample(range(D_TR_IMG.shape[0]), BATCH_SIZE)
    data_batch_img = D_TR_IMG[sample_list]
    data_batch_lab_onehot = D_TR_LAB_onehot[sample_list]

    s_tensor = data_batch_img
    has_prob, ha_prob, ha_value, a_sample, act_entropy = model_WTA.forward(s_tensor)
    action_chosen = torch.squeeze(a_sample, dim=1)
    
    result_diff = torch.abs(action_chosen - data_batch_lab_onehot)
    reward = torch.sum(result_diff, dim=1) * (-0.5) + 1
    accuracy = torch.mean(reward).item()

    reward = reward - 1

    s_batch = s_tensor
    batch_size = s_batch.shape[0]
    # Actor loss
    advantage = reward
    has_prob_batch = has_prob
    ha_prob_batch = ha_prob
    ha_value_batch = ha_value
    ae_batch = act_entropy
    list_len = has_prob_batch.shape[0]
    advantage = advantage.detach().reshape([list_len]) # + args.entropy * ae_batch.reshape([list_len])
    q_has_res = has_prob_batch.expand([1, list_len, model_WTA.dim_has]).permute([1, 2, 0])
    ha_temp = (ha_value_batch - ha_prob_batch).expand([1, list_len, model_WTA.dim_ha]).permute([1, 0, 2])
    weight_target = torch.bmm(q_has_res, ha_temp)
    weight_target[:, 0:model_WTA.dim_ha, :] = 1 * weight_target[:, 0:model_WTA.dim_ha, :] \
                                              + 1 * weight_target[:, 0:model_WTA.dim_ha, :].clone().permute([0, 2, 1])
    weight_target = weight_target * advantage.expand([model_WTA.dim_has, model_WTA.dim_ha, list_len]).permute([2, 0, 1])
    weight_target = torch.mean(weight_target, dim=0) * model_WTA.mask_weight
    bias_target = ha_value_batch * advantage.expand([model_WTA.dim_ha, list_len]).permute([1, 0])
    bias_target = torch.mean(bias_target, dim=0)
    LEARNING_RATE = args.lr
    model_WTA.weight = model_WTA.weight + weight_target * LEARNING_RATE
    model_WTA.bias = model_WTA.bias + bias_target * LEARNING_RATE

    train_accuracy_list.append(accuracy)

    if train_step_i % 100 == 0:             # TEST
        s_tensor = D_TE_IMG
        has_prob, ha_prob, ha_value, a_sample, act_entropy = model_WTA.forward(s_tensor)
        action_prob = has_prob[:, model_WTA.dim_h:model_WTA.dim_ha]

        action_chosen = torch.argmax(action_prob, dim=1)
        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen, num_classes=10)
        test_result_diff = action_chosen_onehot - D_TE_LAB_onehot
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) * (-0.5) + 1
        test_accuracy = torch.mean(test_reward).item()

        train_accuracy_ave = sum(train_accuracy_list) / len(train_accuracy_list)
        train_accuracy_list = []

        record_text = '%9d %8.4f %8.4f' % (train_step_i, train_accuracy_ave, test_accuracy)
        log_text(record_text, test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy_epi_i = train_step_i
            max_accuracy = test_accuracy
            model_WTA.save_model()



# ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
SELECTED_EPI = max_accuracy_epi_i

log_text('SELECTED_EPI' + '  %8d' % SELECTED_EPI)

# -----WEIGHT NOISE------
NOISE_TYPE_LIST = ['gaussian', 'uniform']
for noise_type in NOISE_TYPE_LIST:
    # -----noise param-------
    if noise_type in ['gaussian']:
        noise_param_list = np.arange(0, 1.0, 0.02)
    else:    # if noise_type in ['uniform']:
        noise_param_list = np.arange(0, 4.0, 0.05)
    # -----handle weight-----
    for noise_param in noise_param_list:
        model_WTA.weight = torch.load('./log/' + TIME_STAMP + '_w_best.pt')
        model_WTA.bias = torch.load('./log/' + TIME_STAMP + '_b_best.pt')
        model_WTA.mask_weight = torch.load('./log/' + TIME_STAMP + '_m_best.pt')
        if noise_type == 'gaussian':
            weight_noise = torch.randn(model_WTA.weight.size(), device=model_WTA.dev) * noise_param
            bias_noise = torch.randn(model_WTA.bias.size(), device=model_WTA.dev) * noise_param
        else:  # elif net_noise_type == 'uniform':
            weight_noise = (torch.rand(model_WTA.weight.size(), device=model_WTA.dev) - 0.5) * 2 * noise_param
            bias_noise = (torch.rand(model_WTA.bias.size(), device=model_WTA.dev) - 0.5) * 2 * noise_param
        weight_noise[model_WTA.index_2x, model_WTA.index_2y] = weight_noise[model_WTA.index_1x, model_WTA.index_1y]
        model_WTA.weight += weight_noise * model_WTA.mask_weight
        model_WTA.bias += bias_noise

        has_prob, ha_prob, ha_value, a_sample, act_entropy = model_WTA.forward(D_TE_IMG)
        action_prob = has_prob[:, model_WTA.dim_h:model_WTA.dim_ha]

        action_chosen = torch.argmax(action_prob, dim=1)
        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen, num_classes=10)
        test_result_diff = action_chosen_onehot - D_TE_LAB_onehot
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) * (-0.5) + 1
        test_accuracy = torch.mean(test_reward).item()

        record_text = 'W_' + noise_type + '  %8.4f %8.4f' % (noise_param, test_accuracy)
        log_text(record_text)


# -----INPUT NOISE-------
NOISE_TYPE_LIST = ['gaussian', 'pepper', 'salt', 's&p', 'gaussian&salt']
model_WTA.weight = torch.load('./log/' + TIME_STAMP + '_w_best.pt')
model_WTA.bias = torch.load('./log/' + TIME_STAMP + '_b_best.pt')
model_WTA.mask_weight = torch.load('./log/' + TIME_STAMP + '_m_best.pt')
for noise_type in NOISE_TYPE_LIST:
    # -----noise param-------
    if noise_type in ['gaussian']:
        noise_param_list = np.arange(0, 1.6, 0.05)
    if noise_type in ['pepper', 'salt', 's&p']:
        noise_param_list = np.arange(0, 0.51, 0.02)
    if noise_type in ['gaussian&salt']:
        noise_param_list = np.arange(0, 0.505, 0.005)
    # -----handle data-------
    for noise_param in noise_param_list:
        data_test_img = np.copy(D_TE_IMG.cpu())
        for img_i in range(data_test_img.shape[0]):
            temp_img = data_test_img[img_i, :].reshape([28, 28])
            if noise_type == 'gaussian':
                temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None, clip=True, var=noise_param**2)
            elif noise_type == 'pepper':
                temp_img = skimage.util.random_noise(temp_img, mode='pepper', seed=None, clip=True, amount=noise_param)
            elif noise_type == 'salt':
                temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True, amount=noise_param)
            elif noise_type == 's&p':
                temp_img = skimage.util.random_noise(temp_img, mode='s&p', seed=None, clip=True, amount=noise_param, salt_vs_pepper=0.5)
            elif noise_type == 'gaussian&salt':
                temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None, clip=True, var=0.05**2)
                temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True, amount=noise_param)
            data_test_img[img_i, :] = temp_img.reshape([784])
        data_test_img = torch.from_numpy(data_test_img).to(DEVICE).float()
        # -----test--------------
        has_prob, ha_prob, ha_value, a_sample, act_entropy = model_WTA.forward(data_test_img)
        action_prob = has_prob[:, model_WTA.dim_h:model_WTA.dim_ha]

        action_chosen = torch.argmax(action_prob, dim=1)
        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen, num_classes=10)
        test_result_diff = action_chosen_onehot - D_TE_LAB_onehot
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) * (-0.5) + 1
        test_accuracy = torch.mean(test_reward).item()

        record_text = 'I_' + noise_type + '  %8.4f %8.4f' % (noise_param, test_accuracy)
        log_text(record_text)

print('FINISHED')
