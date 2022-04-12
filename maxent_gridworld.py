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

    wind = 0.0
    trajectory_length = 3*grid_size

    gw = Gridworld.Gridworld(grid_size, wind, discount)

    
    src,dst = gw.generate_schedule(n_schedules)
    c_map = gw.generate_capacity_map(5,25,5)
    path_sol_point,trajectories = gw.solve_congestion_prob(src,dst,n_schedules,c_map)

    feature_matrix = gw.feature_matrix()
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    r = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)


    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(5, 0.01, 20, 200, 0.01, 40)
