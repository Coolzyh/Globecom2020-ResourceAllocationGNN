# implement the Training main function

import matplotlib.pyplot as plt
from BS_brain import Agent
from Environment import *
import pickle
import random
import numpy as np
import tensorflow as tf
from Sim_Config import RL_Config
import os
import keras
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use GPU 0 to run this code

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def main():
    """
    Train the agent
    """
    # number of different trainings settings
    train_num = [1,  2,  3,  4,  5,
                 6,  7,  8,  9]
    # number of each D2D's Compression Features
    num_feedback_set = [16]
    gamma_set = [0.5]
    batch_set = [512]

    # weight for the V2V sum rate
    v2v_weight = 1
    # weight for the V2I sum rate
    v2i_weight = 0.1

    num_train_settings = 1

    # start training
    for train_loop in range(num_train_settings):

        # set the current random seed for training
        train_seed_sequence = 1001
        random.seed(train_seed_sequence)
        np.random.seed(train_seed_sequence)
        tf.set_random_seed(train_seed_sequence)

        # set values for current simulation
        curr_RL_Config = RL_Config()

        train_show_tra = '-----Start the Number -- ' + str(train_num[train_loop]) + ' -- training -----!'
        print(train_show_tra)

        # set key parameters for this train
        num_feedback = num_feedback_set[0]
        gamma = gamma_set[0]
        batch_size = batch_set[0]
        curr_RL_Config.set_train_value(num_feedback, gamma, batch_size, v2v_weight, v2i_weight)

        # start the Environment
        Env = start_env()

        # run the training process
        [Train_Loss,  Reward_Per_Train_Step, Reward_Per_Episode,
        Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean] \
            = run_train(Env, curr_RL_Config)

        # save the train results
        save_flag = save_train_results(Train_Loss, Reward_Per_Train_Step, Reward_Per_Episode,
                                       Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean,
                                       curr_RL_Config, Env)
        if save_flag:
            print('RL Training is finished!')


def start_env():
    # start the environment simulator
    """
    Generate the Environment
    """
    up_lanes = [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

    Env.new_random_game(Env.n_Veh)

    return Env


def run_train(Env, curr_RL_Config):
    # run the training process

    Num_neighbor = Env.n_Neighbor
    Num_d2d = Env.n_Veh
    Num_CH = Env.n_RB
    Num_D2D_feedback = curr_RL_Config.Num_Feedback

    # construct a BS agent
    BS_Agent = Agent(Num_d2d, Num_CH, Num_neighbor, Num_D2D_feedback, Env, curr_RL_Config)
    Num_Episodes = curr_RL_Config.Num_Episodes
    Num_Train_Step = curr_RL_Config.Num_Train_Steps

    # get the train loss
    [Train_Loss,  Reward_Per_Train_Step, Reward_Per_Episode,
    Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean] \
        = BS_Agent.train(Num_Episodes, Num_Train_Step)

    return [Train_Loss,  Reward_Per_Train_Step, Reward_Per_Episode,
            Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean]


def save_train_results(Train_Loss, Reward_Per_Train_Step, Reward_Per_Episode,
                       Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean,
                       curr_rl_config, Env):

    # plot and save the training results

    # get the current training parameter values from curr_rl_config
    Batch_Size = curr_rl_config.Batch_Size
    Num_Train_Step = curr_rl_config.Num_Train_Steps
    Num_Episodes = curr_rl_config.Num_Episodes
    Num_D2D_feedback = curr_rl_config.Num_Feedback
    GAMMA = curr_rl_config.Gamma
    Num_D2D = Env.n_Veh    # number of D2D
    V2I_Weight = curr_rl_config.v2i_weight

    save_flag = False

    # record the Target Q value
    Train_Loss_per_Episode = np.zeros((Num_D2D, Num_Episodes))
    Train_Q_mean_per_Episode = np.zeros((Num_D2D, Num_Episodes))
    Train_Q_max_mean_per_Episode = np.zeros((Num_D2D, Num_Episodes))
    # record the original Q Value
    Orig_Train_Q_mean_per_Episode = np.zeros((Num_D2D, Num_Episodes))
    Orig_Train_Q_max_mean_per_Episode = np.zeros((Num_D2D, Num_Episodes))
    # calculate for each D2D
    for D_loop in range(Num_D2D):
        Train_Loss_per_Episode[D_loop, :] = np.sum(Train_Loss[D_loop, :, :], axis=1) / Num_Train_Step
        Train_Q_mean_per_Episode[D_loop, :] = np.sum(Train_Q_mean[D_loop, :, :], axis=1) / Num_Train_Step
        Train_Q_max_mean_per_Episode[D_loop, :] = np.sum(Train_Q_max_mean[D_loop, :, :], axis=1) / Num_Train_Step
        Orig_Train_Q_mean_per_Episode[D_loop, :] = np.sum(Orig_Train_Q_mean[D_loop, :, :], axis=1) / Num_Train_Step
        Orig_Train_Q_max_mean_per_Episode[D_loop, :] = np.sum(Orig_Train_Q_max_mean[D_loop, :, :], axis=1) / Num_Train_Step

    # save results in their corresponding simulation parameter settings
    curr_sim_set = 'Train-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                   + '-Gamma-' + str(GAMMA) \
                   + '-V2Iweight-' + str(V2I_Weight)

    folder = os.getcwd() + '\\' + curr_sim_set + '\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('Create the new folder in train main ', folder)

    curr_Result_Dir = folder

    # plot loss and Q values for each D2D
    for D_loop in range(Num_D2D):
        # plot the training loss
        x = range(Num_Episodes)
        y = Train_Loss_per_Episode[D_loop, :]
        plt.figure()
        plt.plot(x, y, color='red', label='RL-DNN Train')
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Training Loss")
        plt.grid(True)
        plt.legend()
        Curr_OS = os.name
        if Curr_OS == 'nt':
            print('Current OS is Windows！')
            Fig_Dir = curr_Result_Dir

        Fig_Name = 'D2D-' + str(D_loop) + '-th-Train-LOSS-' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                   + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'D2D-' + str(D_loop) + '-th-Train-LOSS-' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                   + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.eps'
        Fig_Para1 = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para1)

        # plot the Q mean and Q max mean results --- Target Q function
        x = range(Num_Episodes)
        y = Train_Q_mean_per_Episode[D_loop, :]
        y1 = Train_Q_max_mean_per_Episode[D_loop, :]
        plt.figure()
        plt.plot(x, y, color='red', label='mean Target Value')
        plt.plot(x, y1, color='blue', label='max Target Value')
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Q Value")
        plt.grid(True)
        plt.legend()
        # save the figure
        Fig_Name = 'D2D-' + str(D_loop) + '-th-Target-Q-Func-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                   + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'D2D-' + str(D_loop) + '-th-Target-Q-Func-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                    + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.eps'
        Fig_Para1 = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para1)

        # plot the Q mean and Q max mean results  --- Original Q function
        x = range(Num_Episodes)
        y = Orig_Train_Q_mean_per_Episode[D_loop, :]
        y1 = Orig_Train_Q_max_mean_per_Episode[D_loop, :]

        # plot the results
        plt.figure()
        # plt.plot(x, y)
        plt.plot(x, y, color='red', label='Q mean')
        plt.plot(x, y1, color='blue', label='Q-max mean')
        plt.xlabel("Number of Training Episodes")
        plt.ylabel("Q Value")
        # open the grid
        plt.grid(True)
        # plt.title("Q Function in Training")
        plt.legend()
        # save the figure
        Fig_Name = 'D2D-' + str(D_loop) + '-th-Original-Q-Func-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                   + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        # use for latex
        Fig_Name1 = 'D2D-' + str(D_loop) + '-th-Original-Q-Func-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' \
                    + str(Num_Train_Step) + '-Batch-' + str(Batch_Size) + '.eps'
        Fig_Para1 = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para1)

    # plot the training return
    x = range(Num_Episodes)
    y = Reward_Per_Episode
    plt.figure()
    plt.plot(x, y, color='blue', label='DNN-RL')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Return per Episode")
    # open the grid
    plt.grid(True)
    plt.title("Reward per episode in Training")
    plt.legend()
    # save the figure
    Fig_Name = 'Reward-per-Episode-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
               + '-Batch-' + str(Batch_Size) + '.png'
    Fig_Para = Fig_Dir + Fig_Name
    plt.savefig(Fig_Para, dpi=600)
    Fig_Name1 = 'Reward-per-Episode-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
                + '-Batch-' + str(Batch_Size) + '.eps'
    Fig_Para1 = Fig_Dir + Fig_Name1
    plt.savefig(Fig_Para1, dpi=600)

    # save the results to file
    if Curr_OS == 'nt':
        # print('Current OS is Windows！')
        Data_Dir = curr_Result_Dir
    Data_Name = 'Training-Result' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
                + '-Batch-' + str(Batch_Size) + '.pkl'
    Data_Para = Data_Dir + Data_Name
    # open data file
    file_to_open = open(Data_Para, 'wb')
    # write D2D_Sample to data file
    pickle.dump((Train_Loss_per_Episode, Train_Loss,
                 Train_Q_mean, Train_Q_max_mean,
                 Train_Q_mean_per_Episode, Train_Q_max_mean_per_Episode,
                 # record the original Q Value
                 Orig_Train_Q_mean, Orig_Train_Q_max_mean,
                 Orig_Train_Q_mean_per_Episode, Orig_Train_Q_max_mean_per_Episode,
                 Reward_Per_Train_Step, Reward_Per_Episode), file_to_open)
    file_to_open.close()

    save_flag = True

    return save_flag


if __name__ == '__main__':
    main()
