# generate the V2X environment

from __future__ import division
import numpy as np
import random
import math


class RandomGenerate:
    # implement some distributions via python built-in Random module
    def __init__(self):
        print('construct a RandomGenerate class')

    def gauss_one_d(self, mu, sigma, x_size):
        # mu: mean of Gaussian, sigma: variance of Gaussian
        # generate one gaussian sequence with length of x_size
        one_d_gauss = np.zeros(x_size)
        for x_loop in range(x_size):
            one_d_gauss[x_loop] = random.gauss(mu, sigma)

        return one_d_gauss

    def gauss_two_d(self, mu, sigma, x_size, y_size):
        # mu: mean of Gaussian, sigma: variance of Gaussian
        # generate one gaussian array with dimensions of x_size, y_size
        two_d_gauss = np.zeros((x_size, y_size))
        for x_loop in range(x_size):
            for y_loop in range(y_size):
                two_d_gauss[x_loop, y_loop] = random.gauss(mu, sigma)

        return two_d_gauss

    def gauss_three_d(self, mu, sigma, x_size, y_size, z_size):
        # mu: mean of Gaussian, sigma: variance of Gaussian
        # generate 3D gaussian array with dimensions of x_size, y_size, z_size
        three_d_gauss = np.zeros((x_size, y_size, z_size))
        for x_loop in range(x_size):
            for y_loop in range(y_size):
                for z_loop in range(z_size):
                    three_d_gauss[x_loop, y_loop, z_loop] = random.gauss(mu, sigma)

        return three_d_gauss


class V2Vchannels:
    # Simulator of the V2V Channels
    def __init__(self, n_Veh, n_RB):
        # add random generate class
        self.randgen = RandomGenerate
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        # compute the pathloss between any two positions  [size: Num_Veh x Num_Veh ]
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    def update_shadow(self, delta_distance_list):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0:
            # self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            self.Shadow = self.randgen.gauss_two_d(self.randgen, 0, self.shadow_std, self.n_Veh, self.n_Veh)
        else:
            # shadow_temp = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            shadow_temp = self.randgen.gauss_two_d(self.randgen, 0, self.shadow_std, self.n_Veh, self.n_Veh)
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow \
                          + np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) \
                          * shadow_temp

    def update_fast_fading(self):
        mu = 0
        sigma = 1
        real_part = self.randgen.gauss_three_d(self.randgen, mu, sigma, self.n_Veh, self.n_Veh, self.n_RB)
        imag_part = self.randgen.gauss_three_d(self.randgen, mu, sigma, self.n_Veh, self.n_Veh, self.n_RB)
        h = 1/np.sqrt(2) * (real_part + 1j * imag_part)
        self.FastFading = 20 * np.log10(np.abs(h))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) \
                           - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)

        def PL_NLos(d_a, d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1, d2) < 7:
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
            self.ifLOS = False
            self.shadow_std = 4
        return PL


class V2Ichannels:
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_RB):
        # add random generate class
        self.randgen = RandomGenerate
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750/2, 1299/2]
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1, d2)  # change from meters to kilometers
            self.PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000)

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:
            self.Shadow = self.randgen.gauss_one_d(self.randgen, 0, self.shadow_std, self.n_Veh)
        else:
            delta_distance = np.asarray(delta_distance_list)
            shadow_temp = self.randgen.gauss_one_d(self.randgen, 0, self.shadow_std, self.n_Veh)
            self.Shadow = np.exp(-1*(delta_distance/self.Decorrelation_distance))*self.Shadow \
                + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))\
                * shadow_temp

    def update_fast_fading(self):
        mu = 0
        sigma = 1
        real_part = self.randgen.gauss_two_d(self.randgen, mu, sigma, self.n_Veh, self.n_RB)
        imag_part = self.randgen.gauss_two_d(self.randgen, mu, sigma, self.n_Veh, self.n_RB)

        h = 1/np.sqrt(2) * (real_part + 1j * imag_part)
        self.FastFading = 20 * np.log10(np.abs(h))


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    # Environment Simulator: Provide states and rewards to agents.
    # Evolve to new state based on the actions taken by the vehicles.
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height):
        self.timestep = 0.01
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height
        self.vehicles = []
        self.demands = []
        self.V2V_power_dB = 23  # [unit:dBm]
        self.V2I_power_dB = 23  # [unit:dBm]
        self.V2V_power_dB_List = [23, 10, 5]      # the power levels for V2V agent  [unit:dBm]
        self.fixed_v2v_power_index = 1     # prefixed V2V power index: same fixed power selection for each vehicle
        self.sig2_dB = -114  # noise power [unit:dBm]
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10**(self.sig2_dB/10)
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.n_RB = 4     # number of resource block
        self.n_Veh = 4
        self.n_Neighbor = 1
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_Neighbor, self.n_RB)) + self.sig2
        self.n_step = 0
        self.randgen = RandomGenerate

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        for i in range(n):
            ind = random.randrange(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0, self.height)]
            start_direction = 'd'
            self.add_new_vehicles(start_position, start_direction, random.randint(10, 15))
            start_position = [self.up_lanes[ind], random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, random.randint(10, 15))
            start_position = [random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, random.randint(10, 15))
            start_position = [random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, random.randint(10, 15))
        self.V2V_Shadowing = self.randgen.gauss_two_d(self.randgen, 0, 3, len(self.vehicles), len(self.vehicles))
        self.V2I_Shadowing = self.randgen.gauss_one_d(self.randgen, 0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])

    def renew_positions(self):
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        while i < len(self.vehicles):
            # print ('start iteration ', i)
            # print(self.position, len(self.position), self.direction)
            delta_distance = self.vehicles[i].velocity * self.timestep
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <=self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >=self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - ( self.vehicles[i].position[1]- self.left_lanes[j])), self.left_lanes[j] ]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >=self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + ( self.vehicles[i].position[1]- self.right_lanes[j])),self.right_lanes[j] ]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance
            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
            # delete
            # print ('delete ', self.position[i])
                if (self.vehicles[i].direction=='u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0],self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1],self.vehicles[i].position[1]]

            i += 1

    def update_large_fading(self, positions, time_step):
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = time_step * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)

    def update_small_fading(self):
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()

    def renew_neighbor(self):
        # ==========================================
        # update the neighbors of each vehicle.
        # ===========================================
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        Distance = np.zeros((len(self.vehicles), len(self.vehicles)))
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T-z)
        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])  # sort vehicle according to their distances to the target vehicle
            for j in range(self.n_Neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j+1])
            neighbor_range = sort_idx[1:(len(sort_idx) - 2)]
            destination = random.sample(list(neighbor_range), self.n_Neighbor)
            self.vehicles[i].destinations = destination

    def renew_channel(self):
        # ===========================================================================
        # This function updates all the channels including V2V and V2I channels
        # =============================================================================
        positions = [c.position for c in self.vehicles]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(
            len(self.vehicles))
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow

    def renew_channels_fastfading(self):
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        self.renew_channel()
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2Vchannels.FastFading
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - self.V2Ichannels.FastFading

    def compute_reward_with_channel_selection(self, actions_ch_sel):
        # ===================================================
        # --- Compute the rewards with channel selections ---
        # ===================================================
        actions = actions_ch_sel
        power_selection = self.fixed_v2v_power_index*np.ones([self.n_Veh, self.n_Neighbor], dtype='int32')
        V2I_Flag = True
        Interference = np.zeros(self.n_RB)
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i, :])):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i, j]]
                                                     - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                     + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), self.n_Neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_Neighbor))
        Interfence_times = np.zeros((len(self.vehicles), self.n_Neighbor))
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10**(
                        (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                            - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i]
                            + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                if V2I_Flag:
                    if i < self.n_Veh:
                        V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10**(
                                (self.V2I_power_dB - self.V2V_channels_with_fastfading[i][receiver_j][i]
                                    + 2*self.vehAntGain - self.vehNoiseFigure)/10)

                for k in range(j+1, len(indexes)):
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10**(
                            (self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]
                             + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10**(
                            (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]
                             + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j, 0], indexes[j, 1]] += 1
                    Interfence_times[indexes[k, 0], indexes[k, 1]] += 1
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        V2I_Signals = self.V2I_power_dB - self.V2I_channels_abs[0:min(self.n_RB, self.n_Veh)] \
                      + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB, self.n_Veh)]))

        return V2V_Rate, V2I_Rate, Interference

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        V2V_Interference = np.zeros((len(self.vehicles), self.n_Neighbor, self.n_RB)) + self.sig2
        V2I_Flag = True
        if len(actions.shape) == 2:
            channel_selection = actions.copy()
            power_selection = self.fixed_v2v_power_index * np.ones([self.n_Veh, self.n_Neighbor], dtype='int32')
            channel_selection[np.logical_not(self.activate_links)] = -1

            if V2I_Flag:
                # for the i-th RB
                for i in range(self.n_RB):
                    # for the k-th Vehicle
                    for k in range(len(self.vehicles)):
                        # for the m-th neighbor
                        for m in range(len(channel_selection[k, :])):
                            V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB
                                            - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i]
                                            + 2 * self.vehAntGain - self.vehNoiseFigure)/10)

            for i in range(len(self.vehicles)):
                for j in range(len(channel_selection[i, :])):
                    for k in range(len(self.vehicles)):
                        for m in range(len(channel_selection[k, :])):
                            if (i == k) or (channel_selection[i, j] >= 0):
                                continue
                            V2V_Interference[k, m, channel_selection[i, j]] += \
                                10**((self.V2V_power_dB_List[power_selection[i, j]]
                                - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i, j]]
                                + 2*self.vehAntGain - self.vehNoiseFigure)/10)

        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)

    def new_random_game(self, n_Veh=0):
        # make a new game
        self.n_step = 0
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh/4))
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.activate_links = np.ones((self.n_Veh, self.n_Neighbor), dtype='bool')


if __name__ == "__main__":
    up_lanes = [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
