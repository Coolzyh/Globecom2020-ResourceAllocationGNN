# implement the Evaluation main function for the GNN-RL scheme

import matplotlib.pyplot as plt
from BS_brain import Agent
from Environment import *
import pickle
from Sim_Config import RL_Config
import random
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use GPU 0 to run this code


def main():

    """
    Test the trained agent
    """
    # set of different testing seeds
    test_num = [1]
    # set of different D2D Compression Features
    num_feedback_set = [16]
    gamma_set = [0.05]
    # set of different batch sizes
    batch_set = [512]
    # number of different testing seeds
    num_test_settings = 1

    # weight for the V2V sum rate
    v2v_weight = 1
    # weight for the V2I sum
    v2i_weight = 0.1
    # evaluation exploration rate epsilon
    Fixed_Epsilon = 0
    # number of test while evaluating the training process
    num_evaluate_trials = 10
    # parameter setting for evaluating
    num_test_episodes = 10000
    num_test_steps = 100
    opt_flag = False

    # start training
    # run at different Compression Features
    curr_FB = num_feedback_set[0]
    FB_str = '>>>>>>>>>Testing Compression Features = ' + str(curr_FB)  \
             + ' at different random seeds<<<<<<<<<'
    print(FB_str)

    for test_loop in range(num_test_settings):

        # set the current random seed for training
        test_seed_sequence = test_num[test_loop]
        random.seed(test_seed_sequence)
        np.random.seed(test_seed_sequence)
        tf.set_random_seed(test_seed_sequence)

        # set values for current simulation
        curr_RL_Config = RL_Config()

        train_show_tra = '----- Start the Number -- ' + str(test_num[test_loop]) + ' -- Testing -----!'
        print(train_show_tra)

        # set key parameters for this train
        num_feedback = num_feedback_set[0]
        gamma = gamma_set[0]
        batch_size = batch_set[0]
        curr_RL_Config.set_train_value(num_feedback, gamma, batch_size, v2v_weight, v2i_weight)

        # display the parameters settings for current trained model
        curr_RL_Config.display()

        # start the Environment
        Env = start_env()

        # load the trained model
        BS_Agent = load_trained_model(Env, curr_RL_Config)

        # set key parameters for this testing
        curr_RL_Config.set_test_values(num_test_episodes, num_test_steps, opt_flag, v2v_weight, v2i_weight)

        # run the testing process and save the testing results
        save_flag = run_test(curr_RL_Config, BS_Agent, test_seed_sequence, Fixed_Epsilon, num_evaluate_trials)

        # track the testing process
        if save_flag:
            print('RL Testing is finished!')


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


def load_trained_model(Env, curr_RL_Config):
    # load the trained RL-DNN model for testing
    Num_neighbor = Env.n_Neighbor
    Num_d2d = Env.n_Veh
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Num_CH = Env.n_RB
    # construct a BS agent
    BS_Agent = Agent(Num_d2d, Num_CH, Num_neighbor, Num_D2D_feedback, Env, curr_RL_Config)

    # load the Trained model weights
    # Training Parameters
    BATCH_SIZE = curr_RL_Config.Batch_Size
    num_episodes = curr_RL_Config.Num_Episodes
    num_train_steps = curr_RL_Config.Num_Train_Steps
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight

    #  load the trained results according to their corresponding simulation parameter settings
    curr_sim_set = 'Train-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(BATCH_SIZE) \
                   + '-Gamma-' + str(GAMMA) \
                   + '-V2Iweight-' + str(V2I_Weight)

    folder = os.getcwd() + '\\' + curr_sim_set + '\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('Create the new folder in Testing main ', folder)

    model_dir = folder
    model_name = 'Q-Network_model_weights' + '-Episode-' + str(num_episodes) \
                 + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
    model_para = model_dir + model_name

    # save the Target Network's weights in case we need it
    target_model_name = 'Target-Network_model_weights' + '-Episode-' + str(num_episodes) + '-Step-' \
                        + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
    target_model_para = model_dir + target_model_name

    # load Q-Function Network weights
    BS_Agent.brain.model.load_weights(model_para)
    # load Target Network weights
    BS_Agent.brain.target_model.load_weights(target_model_para)

    print('Load the trained model successfully under this setting!')

    return BS_Agent

def run_test(curr_RL_Config, BS_Agent, test_seed_sequence, Fixed_Epsilon, num_evaluate_trials):
    # run the test according to current settings via the trained model
    save_flag = True      # check the saving process
    Num_Run_Episodes = curr_RL_Config.Num_Run_Episodes
    Num_Test_Step = curr_RL_Config.Num_Test_Steps
    Opt_Flag = curr_RL_Config.Opt_Flag
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Batch_Size = curr_RL_Config.Batch_Size
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight
    V2V_Weight = curr_RL_Config.v2v_weight

    # for tracking of the test
    print("-----Current Testing Parameters Settings are: ")
    print('     Number of Compression Features: ', Num_D2D_feedback)
    print('     Discount Factor Gamma: ', GAMMA)
    print('     Optimal Scheme Flag: ', Opt_Flag)
    print('     Batch Size: ', Batch_Size)
    print('     Testing Episodes: ', Num_Run_Episodes)
    print('     Testing Steps per Episode: ', Num_Test_Step)
    print('     Testing Seed: ', test_seed_sequence)
    print('     V2V Rate weight: ', V2V_Weight)
    print('     V2I Rate weight: ', V2I_Weight)
    print('     Exploration Rate EPSILON while evaluation: ', Fixed_Epsilon)
    print('     Number of trials while evaluation: ', num_evaluate_trials)

    if Opt_Flag:

        print('To Run Dist-Dec RL-DNN TEST with Optimal Scheme!')

        # Run with Implementing Optimal Scheme
        [Expect_Return, Reward,
         RA_Expect_Return, RA_Reward,
         Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
         Opt_Per_V2B_Interference] \
            = BS_Agent.evaluate_training_diff_trials(Num_Run_Episodes, Num_Test_Step, Opt_Flag,
                                                     Fixed_Epsilon, num_evaluate_trials)

        Num_Run_Episodes = int(Num_Run_Episodes // 5)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Opt-Evaluate-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main with Opt-scheme', folder)

        Opt_V2I_Sum_Rate = np.sum(Opt_Per_V2I_Rate, axis=2)

        Fig_Dir = folder

        # plot the results
        Num_Run_Episodes = int(Num_Run_Episodes*5)
        x = np.arange(5, Num_Run_Episodes+1, 5)
        y = np.mean(Expect_Return, axis=0)
        y1 = np.mean(RA_Expect_Return, axis=0)
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = np.arange(5, Num_Run_Episodes+1, 5)
        y = np.mean(Expect_Return / Opt_Expect_Return)
        y1 = np.mean(RA_Expect_Return / Opt_Expect_Return)
        plt.figure()
        plt.plot(x, y, color='red', label='DNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Testing Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.legend()
        Fig_Name = 'Opt-Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        Data_Dir = folder
        Data_Name = 'Opt-Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write data
        pickle.dump((Expect_Return, Reward,
                    RA_Expect_Return, RA_Reward,
                    Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
                    Opt_Per_V2B_Interference, Opt_V2I_Sum_Rate), file_to_open)
        file_to_open.close()

        save_flag = True

    else:
        print('To Run Dist-Dec RL-DNN Test without Optimal Scheme!')

        [Evaluated_Opt_Expect_Return,
        Evaluation_Return_per_Episode, Evaluation_Reward_per_Episode,
        RA_Expect_Return, RA_Reward] \
            = BS_Agent.evaluate_training_diff_trials(Num_Run_Episodes, Num_Test_Step, Opt_Flag,
                                                     Fixed_Epsilon, num_evaluate_trials)

        Num_Run_Episodes = Num_Run_Episodes // 5

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Evaluate-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight) \
                       + '-Epsilon-' + str(Fixed_Epsilon)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main without Opt-Scheme ', folder)

        print('------> Testing Results for V2V link are: ')

        Expect_Return = Evaluation_Return_per_Episode
        Reward = Evaluation_Reward_per_Episode

        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The average return of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The average return of RA scheme is ', ave_RA_Return)

        Num_Run_Episodes = Num_Run_Episodes * 5

        Curr_OS = os.name
        # save the results to file
        if Curr_OS == 'nt':
            print('Save testing results！ Current OS is Windows！')
            Data_Dir = folder
        Data_Name1 = 'Ave-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write data
        pickle.dump((Evaluated_Opt_Expect_Return, Expect_Return,
                     RA_Expect_Return), file_to_open)
        file_to_open.close()

        Data_Name = 'Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write data
        pickle.dump((Evaluated_Opt_Expect_Return,
                     Expect_Return, Reward,
                     RA_Expect_Return, RA_Reward), file_to_open)
        file_to_open.close()

        if Curr_OS == 'nt':
            Fig_Dir = folder

        # plot mean return and standard error in training for GNN-RL and Random Action schemes
        Mean_Return_per_Episode = np.mean(Evaluation_Return_per_Episode, axis=0)
        Std_Return = np.std(Evaluation_Return_per_Episode, axis=0)
        Shape_Return = Evaluation_Return_per_Episode.shape
        # standard error
        SE_Return = Std_Return / np.sqrt(Shape_Return[0])
        x_max = Num_Run_Episodes
        my_x_ticks = np.arange(5, x_max+1, 5)
        label_str = 'GNN-RL with ' + '$\epsilon = $' + str(Fixed_Epsilon)
        x = my_x_ticks
        y = Mean_Return_per_Episode
        error = SE_Return
        plt.figure()
        plt.plot(x, y, color='red', marker='x', label=label_str)
        plt.fill_between(x, y - error, y + error, facecolor='green')
        RA_Mean_Return_per_Episode = np.mean(RA_Expect_Return, axis=0)
        RA_Std_Return = np.std(RA_Mean_Return_per_Episode, axis=0)
        RA_Shape_Return = RA_Mean_Return_per_Episode.shape
        # standard error
        RA_SE_Return = RA_Std_Return / np.sqrt(RA_Shape_Return[0])
        error1 = RA_SE_Return
        y1 = RA_Mean_Return_per_Episode
        plt.plot(x, y1, color='blue', marker='^', label='Random Action')
        plt.fill_between(x, y1 - error1, y1 + error1, facecolor='yellow')
        plt.xlabel("Number of Episodes", fontsize=12)
        plt.ylabel("Return per Episode", fontsize=12)
        plt.grid(True)
        plt.xlim(0, x_max)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        Fig_Name = 'Comp-RL-RA-Return-per' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Comp-RL-RA-Return-per' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot mean return and standard error in training
        Mean_Return_per_Episode = np.mean(Evaluation_Return_per_Episode, axis=0)
        Std_Return = np.std(Evaluation_Return_per_Episode, axis=0)
        Shape_Return = Evaluation_Return_per_Episode.shape
        # standard error
        SE_Return = Std_Return / np.sqrt(Shape_Return[0])
        x_max = Num_Run_Episodes
        my_x_ticks = np.arange(5, x_max+1, 5)
        label_str = 'GNN-RL with ' + '$\epsilon = $' + str(Fixed_Epsilon)
        x = my_x_ticks
        y = Mean_Return_per_Episode
        error = SE_Return
        plt.figure()
        plt.plot(x, y, color='red', marker='x', label=label_str)
        plt.fill_between(x, y - error, y + error, facecolor='green')
        plt.xlabel("Number of Episodes", fontsize=12)
        plt.ylabel("Return per Episode", fontsize=12)
        plt.grid(True)
        plt.xlim(0, x_max)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        Fig_Name = 'Marker-Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Marker-Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                    + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot mean return in training
        y = Mean_Return_per_Episode
        plt.figure()
        x = my_x_ticks
        plt.plot(x, y, color='blue')
        plt.xlabel("Number of Episodes", fontsize=11)
        plt.ylabel("Mean Return per Episode", fontsize=11)
        plt.grid(True)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.xlim(0, x_max)
        Fig_Name = 'Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                    + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        save_flag = True

    return save_flag

'''
def run_test(curr_RL_Config, BS_Agent, test_seed_sequence, Fixed_Epsilon, num_evaluate_trials):
    # run the test according to current settings via the trained model
    save_flag = True      # check the saving process
    Num_Run_Episodes = curr_RL_Config.Num_Run_Episodes
    Num_Test_Step = curr_RL_Config.Num_Test_Steps
    Opt_Flag = curr_RL_Config.Opt_Flag
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Batch_Size = curr_RL_Config.Batch_Size
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight
    V2V_Weight = curr_RL_Config.v2v_weight

    # for tracking of the test
    print("-----Current Testing Parameters Settings are: ")
    print('     Number of feedback: ', Num_D2D_feedback)
    print('     Discount Factor Gamma: ', GAMMA)
    print('     Optimal Scheme Flag: ', Opt_Flag)
    print('     Batch Size: ', Batch_Size)
    print('     Testing Episodes: ', Num_Run_Episodes)
    print('     Testing Steps per Episode: ', Num_Test_Step)
    print('     Testing Seed: ', test_seed_sequence)
    print('     V2V Rate weight: ', V2V_Weight)
    print('     V2I Rate weight: ', V2I_Weight)
    print('     Exploration Rate EPSILON while evaluation: ', Fixed_Epsilon)
    print('     Number of trials while evaluation: ', num_evaluate_trials)

    if Opt_Flag:

        print('To Run Dist-Dec RL-DNN TEST with Optimal Scheme!')

        # Run with Implementing Optimal Scheme
        [Expect_Return, Reward,
         RA_Expect_Return, RA_Reward,
         Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
         Opt_Per_V2B_Interference] \
            = BS_Agent.evaluate_training_diff_trials(Num_Run_Episodes, Num_Test_Step, Opt_Flag,
                                                     Fixed_Epsilon, num_evaluate_trials)

        Num_Run_Episodes = int(Num_Run_Episodes // 5)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Opt-Evaluate-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main with Opt-scheme', folder)

        Opt_V2I_Sum_Rate = np.sum(Opt_Per_V2I_Rate, axis=2)

        Fig_Dir = folder

        # plot the results
        Num_Run_Episodes = int(Num_Run_Episodes*5)
        x = np.arange(5, Num_Run_Episodes+1, 5)
        y = np.mean(Expect_Return, axis=0)
        y1 = np.mean(RA_Expect_Return, axis=0)
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = np.arange(5, Num_Run_Episodes+1, 5)
        y = np.mean(Expect_Return / Opt_Expect_Return)
        y1 = np.mean(RA_Expect_Return / Opt_Expect_Return)
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Testing Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.legend()
        Fig_Name = 'Opt-Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        Data_Dir = folder
        Data_Name = 'Opt-Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write data
        pickle.dump((Expect_Return, Reward,
                    RA_Expect_Return, RA_Reward,
                    Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
                    Opt_Per_V2B_Interference, Opt_V2I_Sum_Rate), file_to_open)
        file_to_open.close()

        save_flag = True

    else:
        print('To Run Dist-Dec RL-DNN Test without Optimal Scheme!')

        [Evaluated_Opt_Expect_Return,
        Evaluation_Return_per_Episode, Evaluation_Reward_per_Episode,
        RA_Expect_Return, RA_Reward] \
            = BS_Agent.evaluate_training_diff_trials(Num_Run_Episodes, Num_Test_Step, Opt_Flag,
                                                     Fixed_Epsilon, num_evaluate_trials)

        Num_Run_Episodes = Num_Run_Episodes // 5

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Evaluate-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight) \
                       + '-Epsilon-' + str(Fixed_Epsilon)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main without Opt-Scheme ', folder)

        print('------> Testing Results for V2V link are: ')

        Expect_Return = Evaluation_Return_per_Episode
        Reward = Evaluation_Reward_per_Episode

        # to better evaluate the RL performance   --Mar. 11th 2019
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The average return of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The average return of RA scheme is ', ave_RA_Return)

        Num_Run_Episodes = Num_Run_Episodes * 5

        Curr_OS = os.name
        # save the results to file
        if Curr_OS == 'nt':
            print('Save testing results！ Current OS is Windows！')
            Data_Dir = folder
        Data_Name1 = 'Ave-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'rb')
        # write data
        pickle.dump((Evaluated_Opt_Expect_Return, Expect_Return,
                     RA_Expect_Return), file_to_open)
        file_to_open.close()

        Data_Name = 'Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'rb')
        # write data
        pickle.dump((Evaluated_Opt_Expect_Return,
                     Expect_Return, Reward,
                     RA_Expect_Return, RA_Reward), file_to_open)
        file_to_open.close()

        if Curr_OS == 'nt':
            Fig_Dir = folder

        # plot mean return and standard error in training for GNN-RL and Random Action schemes
        Mean_Return_per_Episode = np.mean(Evaluation_Return_per_Episode, axis=0)
        Std_Return = np.std(Evaluation_Return_per_Episode, axis=0)
        Shape_Return = Evaluation_Return_per_Episode.shape
        # standard error
        SE_Return = Std_Return / np.sqrt(Shape_Return[0])
        x_max = Num_Run_Episodes
        my_x_ticks = np.arange(5, x_max+1, 5)
        label_str = 'GNN-RL with ' + '$\epsilon = $' + str(Fixed_Epsilon)
        x = my_x_ticks
        y = Mean_Return_per_Episode
        error = SE_Return
        plt.figure()
        plt.plot(x, y, color='red', marker='x', label=label_str)
        plt.fill_between(x, y - error, y + error, facecolor='green')
        RA_Mean_Return_per_Episode = np.mean(RA_Expect_Return, axis=0)
        RA_Std_Return = np.std(RA_Mean_Return_per_Episode, axis=0)
        RA_Shape_Return = RA_Mean_Return_per_Episode.shape
        # standard error
        RA_SE_Return = RA_Std_Return / np.sqrt(RA_Shape_Return[0])
        error1 = RA_SE_Return
        y1 = RA_Mean_Return_per_Episode
        plt.plot(x, y1, color='blue', marker='^', label='Random Action')
        plt.fill_between(x, y1 - error1, y1 + error1, facecolor='yellow')
        plt.xlabel("Number of Episodes", fontsize=12)
        plt.ylabel("Return per Episode", fontsize=12)
        plt.grid(True)
        plt.xlim(0, x_max)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        Fig_Name = 'Comp-RL-RA-Return-per' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Comp-RL-RA-Return-per' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot mean return and standard error in training for GNN-RL
        Mean_Return_per_Episode = np.mean(Evaluation_Return_per_Episode, axis=0)
        Std_Return = np.std(Evaluation_Return_per_Episode, axis=0)
        Shape_Return = Evaluation_Return_per_Episode.shape
        # standard error
        SE_Return = Std_Return / np.sqrt(Shape_Return[0])
        x_max = Num_Run_Episodes
        my_x_ticks = np.arange(5, x_max+1, 5)
        label_str = 'GNN-RL with ' + '$\epsilon = $' + str(Fixed_Epsilon)
        x = my_x_ticks
        y = Mean_Return_per_Episode
        error = SE_Return
        plt.figure()
        plt.plot(x, y, color='red', marker='x', label=label_str)
        plt.fill_between(x, y - error, y + error, facecolor='green')
        plt.xlabel("Number of Episodes", fontsize=12)
        plt.ylabel("Return per Episode", fontsize=12)
        plt.grid(True)
        plt.xlim(0, x_max)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        Fig_Name = 'Marker-Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Marker-Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                    + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot mean return in training for GNN-RL
        def smooth(a, WSZ):
            out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
            r = np.arange(1, WSZ - 1, 2)
            start = np.cumsum(a[:WSZ - 1])[::2] / r
            stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
            return np.concatenate((start, out0, stop))

        y = smooth(Mean_Return_per_Episode, 5)
        plt.figure()
        x = my_x_ticks
        plt.plot(x, y, color='blue')
        plt.xlabel("Number of Training Episodes", fontsize=12)
        plt.ylabel("Average Return per Episode", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, x_max)
        Fig_Name = 'Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Evaluated-Return-per-' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                    + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        save_flag = True
    return save_flag
'''

if __name__ == '__main__':
    main()
