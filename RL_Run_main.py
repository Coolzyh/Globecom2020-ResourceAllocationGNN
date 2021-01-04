# implement the Testing main function for the GNN-RL scheme

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
    gamma_set = [0.5]
    # set of different batch sizes
    batch_set = [512]
    # number of different testing seeds
    num_test_settings = 1

    # weight for the V2V sum rate
    v2v_weight = 1
    # weight for the V2I sum rate
    v2i_weight = 0.1

    # parameter setting for testing
    num_test_episodes = 2000
    num_test_steps = 100
    opt_flag = False

    # start testing
    # run at different FB
    curr_FB = num_feedback_set[0]
    FB_str = '>>>>>>>>>Testing FB = ' + str(curr_FB) \
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
        save_flag = run_test(curr_RL_Config, BS_Agent, test_seed_sequence)

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
    # load the trained model for testing
    # parameters to construct a BS Agent object
    Num_neighbor = Env.n_Neighbor
    Num_d2d = Env.n_Veh
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Num_CH = Env.n_RB
    # construct a BS agent
    BS_Agent = Agent(Num_d2d, Num_CH, Num_neighbor, Num_D2D_feedback, Env, curr_RL_Config)

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

    target_model_name = 'Target-Network_model_weights' + '-Episode-' + str(num_episodes) + '-Step-' \
                        + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
    target_model_para = model_dir + target_model_name

    # load Q-Function Network weights
    BS_Agent.brain.model.load_weights(model_para)
    # load Target Network weights
    BS_Agent.brain.target_model.load_weights(target_model_para)

    print('Load the trained model successfully under this setting!')

    return BS_Agent


def run_test(curr_RL_Config, BS_Agent, test_seed_sequence):
    # run the test according to current settings
    save_flag = False      # check the saving process
    # Run the agent in environment via the trained model
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

    # set of V2I rate thresholds
    Test_V2I_Sum_Rate_Threshold = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    Num_V2I_Rate_Threshold = np.size(Test_V2I_Sum_Rate_Threshold)

    if Opt_Flag:

        print('To Run Dist-Dec RL-DNN TEST with Optimal Scheme!')

        # Run with Implementing Optimal Scheme
        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
        Per_V2B_Interference,
        RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
        RA_Per_V2B_Interference,
        Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
        Opt_Per_V2B_Interference] \
            = BS_Agent.test_run(Num_Run_Episodes, Num_Test_Step, Opt_Flag)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Opt-Run-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main with Opt-scheme', folder)

        print('------> Testing Results are: ')
        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)
        ave_Opt_Expect_Return = np.sum(Opt_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of Opt Scheme is ', ave_Opt_Expect_Return)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Opt_Per_V2V_Rate = np.sum(Opt_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of Optimal scheme is ', ave_Opt_Per_V2V_Rate)
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Opt_Per_V2I_Rate = np.sum(Opt_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of Optimal scheme is ', ave_Opt_Per_V2I_Rate)
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode  of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of RA scheme is ', ave_RA_Per_V2I_Rate)
        print('$$$$$$$> Testing Results for V2B Interference control are: ')
        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        ave_Opt_Per_V2B_Interference = np.sum(Opt_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of Optimal scheme is ', ave_Opt_Per_V2B_Interference)
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RA scheme is ', RA_ave_Per_V2B_Interference)

        print('$$$$$$$> Testing Results for V2B Sum Rate control are: ')
        # compute the success ratio of V2I sum rate constraint with give threshold
        V2I_Sum_Rate_Success_Ratio = np.zeros(Num_V2I_Rate_Threshold)
        RA_V2I_Sum_Rate_Success_Ratio = np.zeros(Num_V2I_Rate_Threshold)
        Opt_V2I_Sum_Rate_Success_Ratio = np.zeros(Num_V2I_Rate_Threshold)
        Opt_V2I_Sum_Rate = np.sum(Opt_Per_V2I_Rate, axis=2)
        V2I_Sum_Rate = np.sum(Per_V2I_Rate, axis=2)
        RA_V2I_Sum_Rate = np.sum(RA_Per_V2I_Rate, axis=2)

        for thresh_loop in range(Num_V2I_Rate_Threshold):
            curr_threshold = Test_V2I_Sum_Rate_Threshold[thresh_loop]
            print('Current V2I Sum Rate Threshold = ', curr_threshold)
            Opt_Num_Sucess = np.sum(Opt_V2I_Sum_Rate >= curr_threshold)
            Opt_Success_Ratio = Opt_Num_Sucess/Interfernece_Normalizer
            Opt_V2I_Sum_Rate_Success_Ratio[thresh_loop] = Opt_Success_Ratio
            print('      The average Succeed Ratio of V2I Sum Rate of Optimal scheme is ', Opt_Success_Ratio)
            Num_Sucess = np.sum(V2I_Sum_Rate >= curr_threshold)
            Success_Ratio = Num_Sucess/Interfernece_Normalizer
            V2I_Sum_Rate_Success_Ratio[thresh_loop] = Success_Ratio
            print('      The average Succeed Ratio of V2I Sum Rate of RL scheme is ', Success_Ratio)
            RA_Num_Sucess = np.sum(RA_V2I_Sum_Rate >= curr_threshold)
            RA_Success_Ratio = RA_Num_Sucess/Interfernece_Normalizer
            RA_V2I_Sum_Rate_Success_Ratio[thresh_loop] = RA_Success_Ratio
            print('      The average Succeed Ratio of V2I Sum Rate of RA scheme is ', RA_Success_Ratio)

        # plot the results
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        y2 = Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='blue', label='Optimal Scheme')
        plt.xlabel("Number of Testing Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Curr_OS = os.name
        if Curr_OS == 'nt':
            Fig_Dir = folder
        Fig_Name = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Opt_Expect_Return
        y1 = RA_Expect_Return / Opt_Expect_Return
        y2 = Opt_Expect_Return / Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='blue', label='Optimal Scheme')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # compare RA and GNN-RL
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
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

        # compare normalized RA and GNN-RL
        x = range(Num_Run_Episodes)
        y = Expect_Return / Opt_Expect_Return
        y1 = RA_Expect_Return / Opt_Expect_Return
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
        if Curr_OS == 'nt':
            print('Save Test Results, Current OS is Windows！')
            Data_Dir = folder
        Data_Name = 'Opt-Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write data
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                     Per_V2B_Interference, V2I_Sum_Rate,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference, RA_V2I_Sum_Rate,
                    Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
                    Opt_Per_V2B_Interference, Opt_V2I_Sum_Rate), file_to_open)
        file_to_open.close()

        Data_Name1 = 'Ave-Opt-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write data
        pickle.dump((ave_Opt_Expect_Return, ave_Opt_Per_V2I_Rate, Opt_V2I_Sum_Rate_Success_Ratio,
                     ave_Opt_Per_V2B_Interference, ave_Opt_Per_V2V_Rate,
                     ave_Expected_Return, ave_Per_V2I_Rate, V2I_Sum_Rate_Success_Ratio,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate, RA_V2I_Sum_Rate_Success_Ratio,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        save_flag = True

    else:

        print('To Run Dist-Dec RL-DNN Test without Optimal Scheme!')

        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
         Per_V2B_Interference,
         RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
         RA_Per_V2B_Interference] \
            = BS_Agent.test_run(Num_Run_Episodes, Num_Test_Step, Opt_Flag)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Run-Result' + '-RealFB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) \
                       + '-Seed-' + str(test_seed_sequence) + '-V2Iweight-' + str(V2I_Weight)

        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main without Opt-Scheme ', folder)

        print('------> Testing Results for V2V link are: ')

        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)

        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The average return of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The average return of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RA scheme is ', ave_RA_Per_V2I_Rate)

        print('$$$$$$$> Testing Results for V2B Interference are: ')
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / (Num_Run_Episodes*Num_Test_Step)
        print('      The average V2B interference of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / (Num_Run_Episodes*Num_Test_Step)
        print('      The average V2B interference of RA scheme is ', RA_ave_Per_V2B_Interference)

        # compute the success ratio of V2I sum rate constraint with give threshold
        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        V2I_Sum_Rate_Success_Ratio = np.zeros(Num_V2I_Rate_Threshold)
        RA_V2I_Sum_Rate_Success_Ratio = np.zeros(Num_V2I_Rate_Threshold)
        V2I_Sum_Rate = np.sum(Per_V2I_Rate, axis=2)
        RA_V2I_Sum_Rate = np.sum(RA_Per_V2I_Rate, axis=2)
        print('$$$$$$$> Testing Results for V2B Sum Rate control are: ')

        for thresh_loop in range(Num_V2I_Rate_Threshold):
            curr_threshold = Test_V2I_Sum_Rate_Threshold[thresh_loop]
            print('Current V2I Sum Rate Threshold = ', curr_threshold)
            Num_Sucess = np.sum(V2I_Sum_Rate >= curr_threshold)
            Success_Ratio = Num_Sucess / Interfernece_Normalizer
            V2I_Sum_Rate_Success_Ratio[thresh_loop] = Success_Ratio
            print('      The average Succeed Ratio of V2I Sum Rate of RL scheme is ', Success_Ratio)
            RA_Num_Sucess = np.sum(RA_V2I_Sum_Rate >= curr_threshold)
            RA_Success_Ratio = RA_Num_Sucess / Interfernece_Normalizer
            RA_V2I_Sum_Rate_Success_Ratio[thresh_loop] = RA_Success_Ratio
            print('      The average Succeed Ratio of V2I Sum Rate of RA scheme is ', RA_Success_Ratio)

        # plot the results
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Testing Episodes")
        plt.ylabel("Expected Return")
        plt.grid(True)
        plt.legend()
        Curr_OS = os.name
        if Curr_OS == 'nt':
            Fig_Dir = folder
        Fig_Name = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Expect_Return
        y1 = RA_Expect_Return / Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='GNN-RL')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Testing Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Norm-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Norm-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                    + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        if Curr_OS == 'nt':
            print('Save testing results！ Current OS is Windows！')
            Data_Dir = folder
        Data_Name = 'Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write data
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                    Per_V2B_Interference,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference), file_to_open)
        file_to_open.close()

        Data_Name1 = 'Ave-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write data
        pickle.dump((V2I_Sum_Rate_Success_Ratio, RA_V2I_Sum_Rate_Success_Ratio,
                     ave_Expected_Return, ave_Per_V2I_Rate,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        save_flag = True

    return save_flag


if __name__ == '__main__':
    main()
