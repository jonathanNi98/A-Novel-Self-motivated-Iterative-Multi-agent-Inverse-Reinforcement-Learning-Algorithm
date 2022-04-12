import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.gridworld as Gridworld
import pulp 
def main(grid_size, discount, n_schedules,learning_rate):
    np.random.seed(0)
    wind = 0.3
    trajectory_length = 3*grid_size

    gw = Gridworld.Gridworld(grid_size, wind, discount)
    src,dst = gw.generate_schedule(n_schedules)
    c_map = gw.generate_capacity_map(10,15,20)
    _,path_sol = gw.solve_congestion_prob(src,dst,n_schedules,c_map)
    print(path_sol)
    for i in range(n_schedules):
        xs = [x[0] for x in path_sol[i]]
        ys = [x[1] for x in path_sol[i]]
        plt.plot(xs,ys)
        plt.plot(xs,ys,'or')
    plt.show()
if __name__ == '__main__':
    main(5, 0.01,100,0.01)