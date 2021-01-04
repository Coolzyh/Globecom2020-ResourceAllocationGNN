# set the key parameters for the GNN-RL scheme

import datetime


class RL_Config:
    """
    Define the Config class
    """
    def __init__(self):
        # key parameters for RL training process
        self.Num_Feedback = 16    # number of Compression Features for each D2D
        self.Num_Episodes = 10000     # number of episodes for training
        self.Num_Train_Steps = 20   # number of steps in each Episode
        self.Batch_Size = 256   # size of the mini-batch for replay
        self.Gamma = 0.2     # discount factor in RL
        self.Num_Run_Episodes = 10  # number of episodes for testing
        self.Num_Test_Steps = 50   # number of step in each testing Episode
        self.Opt_Flag = True    # whether run the optimal scheme while testing the trained model
        # add the v2v rate weight
        self.v2v_weight = 1
        # add the v2i rate weight
        self.v2i_weight = 1

    def set_train_value(self, num_feedback, gamma, batch_size, v2v_weight, v2i_weight):
        self.Num_Feedback = num_feedback    # number of Compression Features for each D2D
        self.Gamma = gamma     # discount factor in RL
        self.Batch_Size = batch_size
        # add weight for V2V and V2I links
        self.v2v_weight = v2v_weight
        self.v2i_weight = v2i_weight

    def display(self):
        # track the simulation settings
        current_datetime = datetime.datetime.now()
        print(current_datetime.strftime('%Y/%m/%d %H:%M:%S'))
        print("Current Training Parameters Settings are: ")
        print('Number of Compression Features: ', self.Num_Feedback)
        print('Discount Factor Gamma: ', self.Gamma)
        print('Batch Size: ', self.Batch_Size)
        print('Training Episodes: ', self.Num_Episodes)
        print('Train Steps per Episode: ', self.Num_Train_Steps)

    def set_test_values(self, num_test_episodes, num_test_steps, opt_flag, v2v_weight, v2i_weight):
        # set the key values for testing
        self.Num_Run_Episodes = num_test_episodes
        self.Num_Test_Steps = num_test_steps
        self.Opt_Flag = opt_flag
        # add weight for V2V and V2I links
        self.v2v_weight = v2v_weight
        self.v2i_weight = v2i_weight
