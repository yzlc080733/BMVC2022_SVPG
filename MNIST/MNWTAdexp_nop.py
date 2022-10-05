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
import torch.nn.functional as F
# -----image-------------
import cv2
import skimage
import argparse
parser = argparse.ArgumentParser(description='Description: PLACEHOLDER')
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--num_action', type=int, default=1)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
R_SEED = 2022 + args.rep
torch.manual_seed(R_SEED)
torch.cuda.manual_seed_all(R_SEED)
np.random.seed(R_SEED)
random.seed(R_SEED)
torch.backends.cudnn.deterministic = True
TIME_STAMP = 'MNIST_WTAdexp_%d_rep%d' % (args.num_action, args.rep)


# ~~~~~~~~~~~~~~~~~~~~CONFIGURATION~~~~~~~~~~~~~~~~~
DIM_STATE = 784
DIM_ACTION = 10

TRAIN_STEP_NUM = 20000
BATCH_SIZE = 100
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.001

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
torch.backends.cudnn.benchmark = True

File = open('./log/log_' + TIME_STAMP + '.txt', 'w')


# ~~~~~~~~~~~~~~~~~~~~FUNCTIONS~~~~~~~~~~~~~~~~~~~~~
def show_image(img_array):
    cv2.namedWindow('SHOW_IMAGE')
    cv2.imshow('SHOW_IMAGE', img_array)
    cv2.waitKey(1)

def save_image(file_name, img_array):
    cv2.imwrite(file_name, img_array)


def index2onehot(index_data, dim_num):
    onehot_data = np.zeros([index_data.shape[0], dim_num], dtype=np.float32)
    onehot_data[np.arange(index_data.shape[0]), index_data] = 1
    return onehot_data

def get_batch(dataset_x, dataset_y, batch_size):
    sample_list = random.sample(range(dataset_x.shape[0]), batch_size)
    return dataset_x[sample_list], dataset_y[sample_list]

def log_text(record_text):
    print(record_text)
    File.write(record_text + '\n')
    File.flush()



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

class Net_BP(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Net_BP, self).__init__()
        self.fc1 = nn.Linear(dim_input, 500)
        self.fc2 = nn.Linear(500, dim_output)
        # self.activate1 = nn.ReLU()
        # self.activate1 = nn.Sigmoid()
        self.activate1 = nn.Softmax(dim=2)
        # self.activate2 = nn.ReLU()
        self.activate2 = nn.Softmax(dim=1)
        # self.activate3 = nn.Softmax(dim=1)

        self.optimizer = torch.optim.RMSprop(self.parameters(), LEARNING_RATE)

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(x.shape[0], 50, 10)
        x = self.activate1(x)
        x = x.reshape(x.shape[0], 500)
        x = self.fc2(x)
        x = self.activate2(x)
        # x = self.activate3(x)
        return x

    def calculate(self, input_img, test_flag=False):
        self.output_distribution = self.forward(input_img)
        if test_flag == True:                   # testing -- argmax
            output_argmax = torch.argmax(self.output_distribution, dim=1)
            self.output_sample = torch.zeros_like(self.output_distribution).to(DEVICE)
            self.output_sample[torch.arange(output_argmax.shape[0]), output_argmax] = 1
        else:                                   # training -- sample
            temp_q = torch.distributions.OneHotCategorical(self.output_distribution)
            self.output_sample = temp_q.sample()
        return self.output_sample
    
    def calculate_with_noise(self, input_img, noise_type, noise_param):
        with torch.no_grad():
            for param in self.parameters():
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(DEVICE) * noise_param)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(DEVICE) - 0.5) * 2 * noise_param)
        return self.calculate(input_img, test_flag=True)


    def optimize(self, reward, learning_rate, _):
        # -----Handle reward-----
        reward = reward - 1
        # -----Target------------
        m = torch.distributions.OneHotCategorical(self.output_distribution)
        loss = (-m.log_prob(self.output_sample) * reward).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, name_fix):
        torch.save(self.state_dict(), './log/BPmodel_' + name_fix + '.pt')

    def load_model(self, name_fix):
        self.load_state_dict(torch.load('./log/BPmodel_' + name_fix + '.pt'))



# ~~~~~~~~~~~~~~~~~~~~NETWORK~~~~~~~~~~~~~~~~~~~~~~~
class Net_FC():
    def __init__(self, dim_input, dim_output):
        # -----Shape params------
        self.dim_input = dim_input
        self.dim_action = dim_output
        self.num_hidden = 20
        self.dim_hidden = 10
        self.num_action = args.num_action                 ###
        self.dim_hid = self.num_hidden * self.dim_hidden
        self.dim_act = self.num_action * self.dim_action
        self.dim_hid_act = self.dim_hid + self.dim_act
        self.dim_all = self.dim_hid + self.dim_act + self.dim_input
        # -----W and b-----------
        self.weight = torch.rand([self.dim_all, self.dim_hid_act], dtype=torch.float32).to(DEVICE)
        self.weight[0:self.dim_hid_act, :] = 0.5 * self.weight[0:self.dim_hid_act, :] + 0.5 * self.weight[0:self.dim_hid_act, :].permute([1, 0])
        self.bias = torch.rand([self.dim_hid_act], dtype=torch.float32).to(DEVICE)
        # -----Mask--------------
        self.mask_weight = torch.ones_like(self.weight).to(DEVICE)
        for hid_i in range(self.num_hidden):
            index = hid_i * self.dim_hidden
            self.mask_weight[index:(index+self.dim_hidden), index:(index+self.dim_hidden)] = 0
        for act_i in range(self.num_action):
            index = self.dim_hid + act_i * self.dim_action
            self.mask_weight[index:(index+self.dim_action), index:(index+self.dim_action)] = 0
        self.mask_weight[self.dim_hid:self.dim_hid_act, self.dim_hid:self.dim_hid_act] = 0
        for x_i in range(self.num_action):
            for y_i in range(self.num_hidden):
                if random.random() < 0.4:
                    index_x = self.dim_hid + x_i * self.dim_action
                    index_y = y_i * self.dim_hidden
                    self.mask_weight[index_x:(index_x+self.dim_action), index_y:(index_y+self.dim_hidden)] = 0
                    self.mask_weight[index_y:(index_y+self.dim_hidden), index_x:(index_x+self.dim_action)] = 0
        self.weight = self.weight * self.mask_weight
        # -----Gradient----------
        self.gradient_his = 0
        # -----Reward------------
        self.reward_record = 0
        # -----Function----------
        self.soft_func_dim2 = torch.nn.Softmax(dim=2)
        # -----Spike-------------
        self.spike_response_length = 50
        self.full_spike_time = 200
        spike_res_type = 'dexp'
        if spike_res_type == 'uniform':
            self.spike_response = torch.ones([1, self.spike_response_length], dtype=torch.float32).to(DEVICE) / self.spike_response_length
        if spike_res_type == 'dexp':
            series = torch.arange(self.spike_response_length, dtype=torch.float32).to(DEVICE)
            self.spike_response = torch.exp(series / (-8.0)) - torch.exp(series / (-2.0))
            self.spike_response = self.spike_response / torch.sum(self.spike_response, dim=0)
        # -----PARAMETERS--------
        self.SPIKE_MODE = True
        self.ADD_NOISE_TO_Q = True

    def reset(self):
        pass

    def calculate_batch(self, input_img, test_flag=False):
        # -----init q h a s------
        input_batch_size = input_img.shape[0]
        self.q_hid_act = torch.rand([input_batch_size, self.dim_hid_act], dtype=torch.float32).to(DEVICE)
        self.q_hid_act = self.hid_act_softmax(self.q_hid_act)
        self.q_s = input_img
        # -----q iteration-------
        if self.SPIKE_MODE:
            spike_record = torch.zeros([input_batch_size, self.dim_all, self.full_spike_time+self.spike_response_length]).to(DEVICE)
            for spike_i in range(self.full_spike_time):
                # -----sample------------
                _ = self.hid_act_sample(self.q_hid_act)
                sample_hid_act = self.v_hid_act.expand([self.spike_response_length, input_batch_size, self.dim_hid_act]).permute([1, 2, 0]).clone()
                sample_state = (torch.rand_like(self.q_s).to(DEVICE) < self.q_s).float().expand([self.spike_response_length, input_batch_size, self.dim_input]).permute([1, 2, 0]).clone()
                spike_record[:, 0:self.dim_hid_act, spike_i:(spike_i+self.spike_response_length)] += sample_hid_act * self.spike_response.expand([input_batch_size, self.dim_hid_act, self.spike_response_length])
                spike_record[:, self.dim_hid_act:self.dim_all, spike_i:(spike_i+self.spike_response_length)] += sample_state * self.spike_response.expand([input_batch_size, self.dim_input, self.spike_response_length])
                # -----update q----------
                if spike_i >= (self.spike_response_length - 1):
                    self.q_ext_s = spike_record[:, :, spike_i].clone()
                    temp_q = torch.mm(self.q_ext_s, self.weight) + self.bias.expand([input_batch_size, self.dim_hid_act])
                    temp_target = self.hid_act_softmax(temp_q)
                    self.q_hid_act = temp_target
        else:
            for step_i in range(50):
                self.q_ext_s = torch.cat([self.q_hid_act, self.q_s], dim=1)
                if test_flag == False and self.ADD_NOISE_TO_Q == True:
                    self.q_ext_s += 0.1 * torch.randn(self.q_ext_s.size()).to(DEVICE)
                else:
                    pass
                temp_q = torch.mm(self.q_ext_s, self.weight) + self.bias.expand([input_batch_size, self.dim_hid_act]).clone()
                temp_target = self.hid_act_softmax(temp_q)
                update_distance = torch.mean(torch.abs(self.q_hid_act - temp_target)).item()
                if update_distance < 0.01:       # 0.001 -> 12 steps; 0.01 -> 6 steps
                    break
                self.q_hid_act = temp_target
        self.q_ext_s = torch.cat([self.q_hid_act, self.q_s], dim=1)
        # -----action output-----
        if test_flag == False:
            v_act_output = self.hid_act_sample(self.q_hid_act)  # sample an action for training
        else:
            v_act_output = self.hid_act_argmax(self.q_hid_act)  # get the argmax for testing
        return v_act_output

    def calculate(self, input_img, test_flag=False):
        if test_flag == True:
            input_batch_size = input_img.shape[0]
            v_act_output = torch.zeros([input_batch_size, self.dim_action], dtype=torch.float32).to(DEVICE)
            for sample_i in range(0, input_batch_size, BATCH_SIZE):
                v_act_output[sample_i:(sample_i+BATCH_SIZE), :] = self.calculate_batch(input_img[sample_i:(sample_i+BATCH_SIZE), :], test_flag)
            return v_act_output
        else:
            return self.calculate_batch(input_img, test_flag)

    def calculate_with_noise(self, input_img, noise_type, noise_param):
        if noise_type == 'gaussian':
            self.weight = self.weight + self.mask_weight * torch.randn(self.weight.size()).to(DEVICE) * noise_param
            self.bias.add_(torch.randn(self.bias.size()).to(DEVICE) * noise_param)
        if noise_type == 'uniform':
            self.weight = self.weight + self.mask_weight * (torch.rand(self.weight.size()).to(DEVICE) - 0.5) * 2 * noise_param
            self.bias.add_(torch.rand(self.bias.size()).to(DEVICE) * noise_param)
        return self.calculate(input_img, test_flag=True)

    def optimize(self, reward, learning_rate, label):
        input_batch_size = reward.shape[0]
        #### TODO: STDP-based optimization

        # Re-calculate reward for each action
        result_diff = torch.abs(self.sampled_action - label.expand([self.num_action, input_batch_size, self.dim_action]).clone().permute([1, 0, 2]))
        reward_all = torch.sum(result_diff, dim=2) * (-0.5) + 1
        num_correct = torch.sum(reward_all, dim=1) * 2 - self.num_action
        reward_tuning_factor = torch.exp(num_correct * num_correct / self.num_action * (-1)) * reward + (1 - reward)
        reward = (reward_all * 2 - 2) * reward_tuning_factor.expand([self.num_action, input_batch_size]).clone().permute([1, 0])
        reward_act_vector = reward.expand([self.dim_action, input_batch_size, self.num_action]).clone().permute([1, 2, 0]).reshape([input_batch_size, self.dim_act])
        reward_hid_vector = torch.mean(reward, dim=1).expand([self.dim_hid, input_batch_size]).clone().permute([1, 0])
        reward_hid_act_vector = torch.cat([reward_hid_vector, reward_act_vector], dim=1)
        reward_ext = reward_hid_act_vector.expand([self.dim_all, input_batch_size, self.dim_hid_act]).clone().permute([1, 0, 2])
        reward_ext[:, self.dim_hid:self.dim_hid_act, 0:self.dim_hid] = reward_ext[:, 0:self.dim_hid, self.dim_hid:self.dim_hid_act].permute([0, 2, 1])
        reward_ext2 = reward_hid_act_vector
        # -----Weight------------
        hqa_temp = self.v_hid_act - self.q_hid_act
        q_ext_s_reshape = self.q_ext_s.expand([1, input_batch_size, self.dim_all]).permute([1, 2, 0]).clone()
        hqa_temp_reshape = hqa_temp.expand([1, input_batch_size, self.dim_hid_act]).permute([1, 0, 2]).clone()
        weight_target = torch.bmm(q_ext_s_reshape, hqa_temp_reshape)
        weight_target[:, 0:self.dim_hid_act, :] = 0.5 * weight_target[:, 0:self.dim_hid_act, :] + 0.5 * weight_target[:, 0:self.dim_hid_act, :].permute(0, 2, 1)
        weight_target = torch.mean(weight_target * reward_ext, dim=0) * self.mask_weight
        # -----Bias--------------
        bias_target = torch.mean(reward_ext2 * self.v_hid_act, dim=0)
        # -----Update------------
        #### Method 1: RMSProp
        # target_strengh = torch.sum(weight_target * weight_target) / torch.sum(self.mask_weight)
        # self.gradient_his = self.gradient_his * 0.99 + target_strengh.item() * 0.01
        # current_lr = 0.01 / math.sqrt(self.gradient_his + 1e-9)
        #### Method 2: SGD
        current_lr = 0.1
        #### --------------------
        self.weight = torch.clone(self.weight + current_lr * weight_target)
        self.bias = torch.clone(self.bias + current_lr * bias_target)

    def hid_act_softmax(self, q_hid_act):
        input_batch_size = q_hid_act.shape[0]
        # -----hidden------------
        q_hid = q_hid_act[:, 0:(self.dim_hid)]
        q_hid_reshape = q_hid.reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        q_hid_reshape_soft = self.soft_func_dim2(q_hid_reshape)
        q_hid_soft = q_hid_reshape_soft.reshape([input_batch_size, self.dim_hid])
        # -----action------------
        q_act = q_hid_act[:, self.dim_hid:(self.dim_hid_act)]
        q_act_reshape = q_act.reshape([input_batch_size, self.num_action, self.dim_action])
        q_act_reshape_soft = self.soft_func_dim2(q_act_reshape)
        q_act_soft = q_act_reshape_soft.reshape([input_batch_size, self.dim_act])
        # -----concat------------
        q_hid_act_soft = torch.cat([q_hid_soft, q_act_soft], dim=1)
        return q_hid_act_soft

    def hid_act_sample(self, q_hid_act):
        v_hid_act = torch.zeros_like(q_hid_act).to(DEVICE)             # sample one-hot vectors according to q_hid_act
        input_batch_size = q_hid_act.shape[0]
        # -----hidden------------
        q_hid = q_hid_act[:, 0:self.dim_hid]
        q_hid_reshape = q_hid.reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        temp_q1 = torch.distributions.OneHotCategorical(q_hid_reshape)
        temp_q_sample = temp_q1.sample()
        v_hid_act[:, 0:self.dim_hid] = temp_q_sample.reshape([input_batch_size, self.dim_hid])
        # -----action------------
        q_act = q_hid_act[:, self.dim_hid:self.dim_hid_act]
        q_act_reshape = q_act.reshape([input_batch_size, self.num_action, self.dim_action])
        temp_q = torch.distributions.OneHotCategorical(q_act_reshape)
        temp_q_sample = temp_q.sample()
        v_hid_act[:, self.dim_hid:self.dim_hid_act] = temp_q_sample.reshape([input_batch_size, self.dim_act])
        # -----vote action-------
        v_act_sum = torch.sum(temp_q_sample, dim=1)
        v_act_argmax = torch.argmax(v_act_sum, dim=1)
        v_act_output = torch.zeros([input_batch_size, self.dim_action], dtype=torch.float32).to(DEVICE)
        v_act_output[torch.arange(input_batch_size), v_act_argmax] = 1
        ## v_act_output = temp_q_sample[:, 1, :]          # test if one of the actions get right
        # -----record------------
        self.v_hid_act = v_hid_act                          # the result sample (hid_act)
        self.sampled_action = temp_q_sample                 # the result sample (act)
        return v_act_output                                 # the result sample (vote result)

    def hid_act_argmax(self, q_hid_act):
        input_batch_size = q_hid_act.shape[0]
        # -----multiple action---
        q_act = q_hid_act[:, self.dim_hid:self.dim_hid_act]
        q_act_reshape = q_act.reshape([input_batch_size, self.num_action, self.dim_action])
        temp_q = torch.argmax(q_act_reshape, dim=2)
        temp_q_onehot = torch.nn.functional.one_hot(temp_q, num_classes=self.dim_action)
        # -----vote--------------
        v_act_sum = torch.sum(temp_q_onehot, dim=1)
        v_act_argmax = torch.argmax(v_act_sum, dim=1)
        v_act_output = torch.zeros([input_batch_size, self.dim_action], dtype=torch.float32).to(DEVICE)
        v_act_output[torch.arange(input_batch_size), v_act_argmax] = 1
        return v_act_output

    def save_model(self, name_fix):
        w_name = './log/' + TIME_STAMP + '_w_best.pt'
        b_name = './log/' + TIME_STAMP + '_b_best.pt'
        m_name = './log/' + TIME_STAMP + '_m_best.pt'
        torch.save(self.weight, w_name)
        torch.save(self.bias, b_name)
        torch.save(self.mask_weight, m_name)

    def load_model(self, name_fix):
        w_name = './log/' + TIME_STAMP + '_w_best.pt'
        b_name = './log/' + TIME_STAMP + '_b_best.pt'
        m_name = './log/' + TIME_STAMP + '_m_best.pt'
        self.weight = torch.load(w_name)
        self.bias = torch.load(b_name)
        self.mask_weight = torch.load(m_name)



# ~~~~~~~~~~~~~~~~~~~~MAIN LOOP~~~~~~~~~~~~~~~~~~~~~
# Net = Net_BP(DIM_STATE, DIM_ACTION).cuda(DEVICE)
Net = Net_FC(DIM_STATE, DIM_ACTION)

test_accuracy = 0
max_accuracy = 0
max_accuracy_epi_i = 0

# TRAIN_STEP_NUM = 0      # Uncomment to move on to testing
for train_step_i in range(TRAIN_STEP_NUM):
    data_batch_img, data_batch_lab_onehot = get_batch(D_TR_IMG, D_TR_LAB_onehot, BATCH_SIZE)
    result = Net.calculate(data_batch_img, test_flag=False)
    result_diff = result - data_batch_lab_onehot
    reward = torch.sum(torch.abs(result_diff), dim=1) * (-0.5) + 1
    accuracy = torch.mean(reward).item()
    Net.optimize(reward, LEARNING_RATE, data_batch_lab_onehot)
    if train_step_i % 100 == 0:
        test_result = Net.calculate(D_TE_IMG, test_flag=True)
        test_result_diff = test_result - D_TE_LAB_onehot
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) * (-0.5) + 1
        test_accuracy = torch.mean(test_reward).item()
        record_text = '%9d %8.4f %8.4f' % (train_step_i, accuracy, test_accuracy)
        log_text(record_text)
        if test_accuracy > max_accuracy:
            max_accuracy_epi_i = train_step_i
            max_accuracy = test_accuracy
            # Net.save_model('%08d' % train_step_i)
            Net.save_model('best')



# ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
SELECTED_EPI = max_accuracy_epi_i


log_text('SELECTED_EPI' + '  %8d' % SELECTED_EPI)

# -----WEIGHT NOISE------
NOISE_TYPE_LIST = ['gaussian', 'uniform']
data_test_img = D_TE_IMG
data_test_lab = D_TE_LAB_onehot
for noise_type in NOISE_TYPE_LIST:
    # -----noise param-------
    if noise_type in ['gaussian']:
        noise_param_list = np.arange(0, 1.0, 0.02)
    if noise_type in ['uniform']:
        noise_param_list = np.arange(0, 4.0, 0.05)
    # -----handle weight-----
    for noise_param in noise_param_list:
        # Net.load_model('%08d' % SELECTED_EPI)
        Net.load_model('best')
        # -----test--------------
        test_result = Net.calculate_with_noise(data_test_img, noise_type, noise_param)
        test_result_diff = test_result - data_test_lab
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) * (-0.5) + 1
        # test_reward = test_reward.float()
        test_accuracy = torch.mean(test_reward).item()
        record_text = 'W_' + noise_type + '  %8.4f %8.4f' % (noise_param, test_accuracy)
        log_text(record_text)



# -----INPUT NOISE-------
NOISE_TYPE_LIST = ['gaussian', 'pepper', 'salt', 's&p', 'gaussian&salt']
Net.load_model('best')
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
        data_test_lab = D_TE_LAB_onehot
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
        # -----test--------------
        test_result = Net.calculate(torch.from_numpy(data_test_img).to(DEVICE), test_flag=True)
        test_result_diff = test_result - data_test_lab
        test_reward = torch.sum(torch.abs(test_result_diff), dim=1) == 0
        test_reward = test_reward.float()
        test_accuracy = torch.mean(test_reward).item()
        record_text = 'I_' + noise_type + '  %8.4f %8.4f' % (noise_param, test_accuracy)
        log_text(record_text)



print('FINISHED')
