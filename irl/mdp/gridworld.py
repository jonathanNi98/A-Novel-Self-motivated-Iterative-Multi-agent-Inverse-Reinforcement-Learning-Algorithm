"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn
import pulp as pulp
import matplotlib.pyplot as plt

class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (int(i % self.grid_size), int(i // self.grid_size))

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)
    
    def generate_schedule(self, n_schedules):
        """
        Args:
            N:numver of schedules
        Generate N (src,dst) pairs
        """
        n=0
        src = np.zeros([n_schedules,2])
        dst = np.zeros([n_schedules,2])
        while n < n_schedules:
            src_t = np.random.randint(0,self.grid_size,size =(1,2))
            dst_t = np.random.randint(0,self.grid_size,size =(1,2))
            if not np.array_equal(src_t,dst_t):
                src[n] = src_t
                dst[n] = dst_t
                n+=1
        return src.astype(int),dst.astype(int)

    def is_valid_cell(self,x, y):
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False

        return True

    def find_paths_util(self,source, destination, visited, path, paths,max_path_length):

        if np.array_equal(source, destination):
            paths.append(path[:])  # append copy of current path
            return

          # mark current cell as visited
        N = self.grid_size
        if len(path) > max_path_length:
            return
        x, y = source
        visited[x][y] = True
          # if current cell is a valid and open cell, 
        if self.is_valid_cell(x, y):
            # Using Breadth First Search on path extension in all direction

            # go right (x, y) --> (x + 1, y)
            if x + 1 < N and (not visited[x + 1][y]):
                path.append(self.point_to_int((x + 1, y)))
                self.find_paths_util((x + 1, y), destination, visited, path, paths,max_path_length)
                path.pop()

            # go left (x, y) --> (x - 1, y)
            if x - 1 >= 0 and (not visited[x - 1][y]):
                path.append(self.point_to_int((x - 1, y)))
                self.find_paths_util((x - 1, y), destination, visited, path, paths,max_path_length)
                path.pop()
            # go up (x, y) --> (x, y + 1)
            if y + 1 < N and (not visited[x][y + 1]):
                path.append(self.point_to_int((x, y + 1)))
                self.find_paths_util((x, y + 1), destination, visited, path, paths,max_path_length)
                path.pop()

            # go down (x, y) --> (x, y - 1)
            if y - 1 >= 0 and (not visited[x][y - 1]):
                path.append(self.point_to_int((x, y - 1)))
                self.find_paths_util((x, y - 1), destination, visited, path, paths,max_path_length)
                path.pop()

        # Unmark current cell as visited
        visited[x][y] = False
    
        return paths
    def find_paths(self,source, destination):
        """ Sets up and searches for paths"""
        N = self.grid_size # size of Maze is N x N

        # 2D matrix to keep track of cells involved in current path
        visited = np.full((N,N),False)

        path = [self.point_to_int(source)]
        paths = []
        paths = self.find_paths_util(source, destination, visited, path, paths, 2 * (self.grid_size-1) )
        return paths
    def paths_to_vector(self,paths):
        vec = np.zeros((len(paths),(self.grid_size**2)))
        for j in range(len(paths)):
            for i in range(len(paths[j])):
                vec[j,paths[j][i]] = 1
        return vec
    def solve_congestion_prob(self,src,dst,n_schedules,capacity_map):
        paths = []
        vec = []
        for i in range(n_schedules):
            paths.append(self.find_paths(src[i], dst[i]))
            vec.append(self.paths_to_vector(paths[i]))
        my_lp_problem = pulp.LpProblem("My LP Problem", pulp.LpMinimize)
        x = pulp.LpVariable.dicts('x', ((i,j) for i in range(n_schedules) for j in range(200)), cat='Binary')
        C = pulp.LpVariable('C', lowBound=0 , cat='Integer')
        my_lp_problem += C
        for k in range(n_schedules):
            my_lp_problem += pulp.lpSum(x[k,i] for i in range(len(vec[k]))) ==1
        for j in range(self.grid_size ** 2):
            my_lp_problem += pulp.lpSum(pulp.lpSum(x[k,i]*vec[k][i,j]
for i in range(vec[k].shape[0]))for k in range(n_schedules)) <= C + capacity_map[j]
        my_lp_problem.solve()
        
        x_sol = np.array([[x[i,j].varValue for j in range(200)]for i in range(n_schedules)])
        path_sol = []
        for k in range(n_schedules):
            for i in range(len(vec[k])):
                if x_sol[k][i]!=0:
                    path_sol.append((np.multiply(x_sol[k][i],paths[k][i])).astype(int))
        path_sol_0 = np.array(path_sol,dtype = object)
        path_sol_point = []
        for i in range(len(path_sol_0)):
            temp = []
            for j in range(len(path_sol_0[i])):
                temp.append(self.int_to_point(path_sol_0[i][j]))
            path_sol_point.append(temp)
        path_sol_point = np.array(path_sol_point,dtype = object)
        return path_sol_point,path_sol

    def generate_capacity_map(self,low,mid,high):
        capacity_map = np.zeros((5,5))
        capacity_map[2,2] = high
        for (i,j) in [[1,1],[1,2],[1,3],[2,1],[2,3],[3,1],[3,2],[3,3]]:
            capacity_map[i,j] = mid
        
        for (i,j) in [[0,0],[0,1],[0,2],[0,3],[0,4],[2,0],[2,4],[3,0],[3,4],[4,0],[4,1],[4,2],[4,3],[4,4],[1,0],[1,4]]:
            capacity_map[i,j] = low
        return capacity_map.flatten()
    