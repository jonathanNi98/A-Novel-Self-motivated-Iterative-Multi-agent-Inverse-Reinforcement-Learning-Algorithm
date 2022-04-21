"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.gridworld as Gridworld
import pulp 
from matplotlib import colors

def main(grid_size, discount, n_trajectories, epochs, learning_rate , n_schedules):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """
    congestion_map_sum = np.zeros(grid_size**2)
    policy_congestion_map_sum = np.zeros(grid_size**2)
    policy_congestion_map_0_sum = np.zeros(grid_size**2)
    policy_congestion_map_1_sum = np.zeros(grid_size**2)
    for p in range(5):
    
        wind = 0.0
        trajectory_length = 3*grid_size

        gw = Gridworld.Gridworld(grid_size, wind, discount)


        src,dst = gw.generate_schedule(n_schedules,0.3)
        c_map = gw.generate_capacity_map(5,10,30)
        congestion_map,trajectories = gw.solve_congestion_prob(src,dst,n_schedules,c_map)

        feature_matrix = gw.feature_matrix()
        ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
        r = maxent.irl(feature_matrix, gw.n_actions, discount,
            gw.transition_probability, trajectories, epochs, learning_rate)

        #value iterarion
        r = r.reshape((grid_size, grid_size))
        r_0 = -np.ones([5,5])
        V_1 = gw.value_iteration(15,r,credit = 1)
        V_1 = V_1.reshape([5,5])
        V_2 = gw.value_iteration(15,r_0,credit = 1)
        V_2 = V_2.reshape([5,5])

        plt.pcolor(r)
        plt.colorbar()
        plt.show()



        V = np.zeros([grid_size**2,grid_size**2])
        for i in range(grid_size**2):
            V[i] = gw.value_iteration(i,-np.exp(r),credit = 1)
        policy_paths = []
        for i in range(n_schedules):
            V_temp = V[gw.point_to_int(dst[i])].reshape(grid_size,grid_size).T
            policy_paths.append(gw.policy_path(gw.point_to_int(src[i]),gw.point_to_int(dst[i]),-np.exp(r),V_temp))
        policy_congestion_map = gw.congestion_heatmap(policy_paths,c_map)


        #greedy policy
        V_0 = np.zeros([grid_size**2,grid_size**2])
        for i in range(grid_size**2):
            V_0[i] = gw.value_iteration(i,r_0,credit = 10)
        policy_paths_0 = []
        for i in range(n_schedules):
            V_temp = V_0[gw.point_to_int(dst[i])].reshape(grid_size,grid_size).T
            policy_paths_0.append(gw.policy_path(gw.point_to_int(src[i]),gw.point_to_int(dst[i]),r_0,V_temp))
        policy_congestion_map_0 = gw.congestion_heatmap(policy_paths_0,c_map)

        V_1 = np.zeros([grid_size**2,grid_size**2])
        c_map_0 = c_map.reshape(grid_size,grid_size)
        for i in range(grid_size**2):
            V_1[i] = gw.value_iteration(i,-1/c_map_0,credit = 10)
        policy_paths_1 = []
        for i in range(n_schedules):
            V_temp = V_1[gw.point_to_int(dst[i])].reshape(grid_size,grid_size).T
            policy_paths_1.append(gw.policy_path(gw.point_to_int(src[i]),gw.point_to_int(dst[i]),-1/c_map_0,V_temp))
        policy_congestion_map_1 = gw.congestion_heatmap(policy_paths_1,c_map)


        fig, axs = plt.subplots(2,2)
        fig.suptitle('Congestion percentage')
        pcm = axs[0,0].pcolor(congestion_map)
        axs[0,0].set_title('Linear programming')
        pcm = axs[0,1].pcolor(policy_congestion_map)
        axs[0,1].set_title('SMIRL')
        pcm = axs[1,0].pcolor(policy_congestion_map_0)
        axs[1,0].set_title('no congestion fee')
        pcm = axs[1,1].pcolor(policy_congestion_map_1)
        axs[1,1].set_title('naive congestion fee')
        axs[0,0].axes.get_xaxis().set_visible(False)
        axs[0,0].axes.get_yaxis().set_visible(False)
        axs[0,1].axes.get_xaxis().set_visible(False)
        axs[0,1].axes.get_yaxis().set_visible(False)
        axs[1,0].axes.get_xaxis().set_visible(False)
        axs[1,0].axes.get_yaxis().set_visible(False)
        axs[1,1].axes.get_xaxis().set_visible(False)
        axs[1,1].axes.get_yaxis().set_visible(False)
        axs[1,0].axes.get_xaxis().set_visible(False)
        axs[1,0].axes.get_yaxis().set_visible(False)
        fig.colorbar(pcm, ax = axs[:2,:2])
        plt.show()


        congestion_map = congestion_map.reshape(grid_size ** 2)
        policy_congestion_map = policy_congestion_map.reshape(grid_size ** 2)
        policy_congestion_map_0 = policy_congestion_map_0.reshape(grid_size ** 2)
        policy_congestion_map_1 = policy_congestion_map_1.reshape(grid_size ** 2)

        br1 = np.arange(0,25)
        barWidth = 0.2
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        plt.bar(br1,congestion_map,color = 'red',width = barWidth)
        plt.bar(br2,policy_congestion_map, color = 'blue',width = barWidth)
        plt.bar(br3,policy_congestion_map_0, color = 'green',width = barWidth)
        plt.bar(br4,policy_congestion_map_1, color = 'yellow',width = barWidth)
        plt.show()


        plt.plot(np.sort(congestion_map),color = 'red',label = 'lp')
        plt.plot(np.sort(policy_congestion_map), color = 'blue',label = 'SMIRL')
        plt.plot(np.sort(policy_congestion_map_0), color = 'green',label = 'no congestion fee')
        plt.plot(np.sort(policy_congestion_map_1), color = 'cyan', label = 'naive congestion fee')
        plt.title('sorted congestion percentage of each node')
        plt.ylabel('congestion percentage')
        plt.legend()
        plt.show()
        
        congestion_map_sum += congestion_map
        policy_congestion_map_sum += policy_congestion_map
        policy_congestion_map_0_sum += policy_congestion_map_0
        policy_congestion_map_1_sum += policy_congestion_map_1
        
    plt.plot(np.sort(congestion_map_sum),color = 'red',label = 'lp')
    plt.plot(np.sort(policy_congestion_map_sum), color = 'blue',label = 'SMIRL')
    plt.plot(np.sort(policy_congestion_map_0_sum), color = 'green',label = 'no congestion fee')
    plt.plot(np.sort(policy_congestion_map_1_sum), color = 'cyan', label = 'naive congestion fee')
    plt.title('sorted congestion percentage of each node(sum)')
    plt.ylabel('congestion percentage')
    plt.legend()
    plt.show()
    
    congestion_map_sum = congestion_map_sum.reshape(grid_size,grid_size)
    policy_congestion_map_sum = policy_congestion_map_sum.reshape(grid_size,grid_size)
    policy_congestion_map_0_sum = policy_congestion_map_0_sum.reshape(grid_size,grid_size)
    policy_congestion_map_1_sum = policy_congestion_map_1_sum.reshape(grid_size,grid_size)
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Congestion percentage')
    pcm = axs[0,0].pcolor(congestion_map_sum)
    axs[0,0].set_title('Linear programming')
    pcm = axs[0,1].pcolor(policy_congestion_map_sum)
    axs[0,1].set_title('SMIRL')
    pcm = axs[1,0].pcolor(policy_congestion_map_0_sum)
    axs[1,0].set_title('no congestion fee')
    pcm = axs[1,1].pcolor(policy_congestion_map_1_sum)
    axs[1,1].set_title('naive congestion fee')
    axs[0,0].axes.get_xaxis().set_visible(False)
    axs[0,0].axes.get_yaxis().set_visible(False)
    axs[0,1].axes.get_xaxis().set_visible(False)
    axs[0,1].axes.get_yaxis().set_visible(False)
    axs[1,0].axes.get_xaxis().set_visible(False)
    axs[1,0].axes.get_yaxis().set_visible(False)
    axs[1,1].axes.get_xaxis().set_visible(False)
    axs[1,1].axes.get_yaxis().set_visible(False)
    axs[1,0].axes.get_xaxis().set_visible(False)
    axs[1,0].axes.get_yaxis().set_visible(False)
    fig.colorbar(pcm, ax = axs[:2,:2])
    plt.show()
if __name__ == '__main__':
    main(5, 0.01, 20, 200, 0.01, 52)