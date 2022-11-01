import numpy as np
class OW(object):
    def __init__(self,num_agents = 20, seed=None, max_r = 100):
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.state = np.ones(num_agents).astype(int)
        self.terminal_state = 13
        self.num_agents = num_agents
        self.max_r = max_r
        self.time = np.zeros(num_agents).astype(int)



    def reset(self):
        self.state = (np.ones(self.num_agents)*8).astype(int)
        self.time = np.zeros(self.num_agents).astype(int)
        self.done = np.zeros(self.num_agents)
        return np.copy(self.state),np.copy(self.time)
    
    def reset_0(self):
        self.state = np.ones(self.num_agents).astype(int)
        self.time = np.arange(self.num_agents).astype(int)
        self.done = np.zeros(self.num_agents)
        return np.copy(self.state),np.copy(self.time)
    
    def time_step(self,reward):
        self.time -= reward
        for i in range(self.num_agents):
            if self.time[i] >= self.max_r:
                self.time[i] = self.max_r-1
                self.time.astype(int)
        return np.copy(self.time)


    def state_action_space(self, state):
        #0:move left ; 1 :move right diagnol; 2: move down; 3:move left diagonal
        if state == 1:
            action_space = [[0,2,0],[1,7,5],[2,6,4]]
        if state == 2:
            action_space = [[0,3,1],[1,8,7],[2,7,6]]
        if state == 3:
            action_space = [[0,4,2],[1,4,3],[2,8,8]]
        if state == 4:
            action_space = [[0,5,3],[1,5,3],[2,9,9]]
        if state == 5:
            action_space = [[0,9,10],[1,9,10],[2,9,10]]
        if state == 6:
            action_space = [[0,7,11],[1,10,14],[2,7,11]]
        if state == 7:
            action_space = [[0,8,12],[1,11,16],[2,10,15]]
        if state == 8:
            action_space = [[0,9,13],[1,12,18],[2,12,18]]
        if state == 9:
            action_space = [[0,13,20],[1,13,20],[2,12,19]]
        if state == 10:
            action_space = [[0,11,21],[1,11,21],[2,11,21]]
        if state == 11:
            action_space = [[0,12,22],[1,12,22],[2,12,22]]
        if state == 12:
            action_space = [[0,13,23],[1,13,23],[2,13,23]]
        if state == 13:
            action_space = [[0,13,24],[1,13,24],[2,13,24]]
        action_space = np.array(action_space)
        return action_space


    def state_transition_func(self, state, action):
        next_state = np.copy(state)
        count = np.zeros(25)

        for j in range(len(state)):
            action_space = self.state_action_space(state[j])
            for i in range(len(action_space)):
                if action[j] == action_space[i][0]:
                    next_state[j] = action_space[i][1]
                    if state[j] == 1 and next_state[j] == 2:
                        count[0] +=1
                    if state[j] == 2 and next_state[j] == 3:
                        count[1] +=1
                    if state[j] == 3 and next_state[j] == 4:
                        count[2] +=1
                    if state[j] == 4 and next_state[j] == 5:
                        count[3] +=1
                    if state[j] == 1 and next_state[j] == 6:
                        count[4] +=1
                    if state[j] == 1 and next_state[j] == 7:
                        count[5] +=1
                    if state[j] == 2 and next_state[j] == 7:
                        count[6] +=1
                    if state[j] == 2 and next_state[j] == 8:
                        count[7] +=1   
                    if state[j] == 3 and next_state[j] == 9:
                        count[8] +=1
                    if state[j] == 4 and next_state[j] == 9:
                        count[9] +=1
                    if state[j] == 5 and next_state[j] == 9:
                        count[10] +=1
                    if state[j] == 6 and next_state[j] == 7:
                        count[11] +=1
                    if state[j] == 7 and next_state[j] == 8:
                        count[12] +=1
                    if state[j] == 8 and next_state[j] == 9:
                        count[13] +=1
                    if state[j] == 6 and next_state[j] == 10:
                        count[14] +=1
                    if state[j] == 7 and next_state[j] == 10:
                        count[15] +=1
                    if state[j] == 7 and next_state[j] == 11:
                        count[16] +=1
                    if state[j] == 8 and next_state[j] == 11:
                        count[17] +=1
                    if state[j] == 8 and next_state[j] == 12:
                        count[18] +=1
                    if state[j] == 9 and next_state[j] == 12:
                        count[19] +=1
                    if state[j] == 9 and next_state[j] == 13:
                        count[20] +=1
                    if state[j] == 10 and next_state[j] == 11:
                        count[21] +=1
                    if state[j] == 11 and next_state[j] == 12:
                        count[22] +=1
                    if state[j] == 12 and next_state[j] == 13:
                        count[23] +=1

        next_state.astype(int)
        return next_state, count
    

    def reward_funtion(self,state,action):
        _,count = self.state_transition_func(state,action)
        count[17]=20000
        count[22]=20000
        count[18]=45
        count[20]=45
        count[13] *= 40
        count[23] *= 40
        count[19] = 1
        N = len(action)
        reward = np.zeros(N)

        for i in range(N):
            action_space = self.state_action_space(state[i])
            if action_space[action[i]][2] ==24:
                reward[i] = 0
            else:
                reward[i] = count[action_space[action[i]][2]]

        return -reward
    
    def step(self, action):
        assert len(action) == len(self.state)
        state = np.copy(self.state)
        reward = self.reward_funtion(self.state,action)
        self.state, count = self.state_transition_func(self.state, action)
        #define reward function and time

        time = self.time_step(reward.astype(int))
        '''if self.state == self.terminal_state:
            reward = 10.0
            done = True
        else:
            done = False
            if self.rng.random() < 0.5:
                reward = -1.0
            else:
                reward = -2.0'''
        for i in range(self.num_agents):
            if self.state[i] == 13:
                self.done[i] = 1
        #return a_
        mean_action = np.zeros([self.num_agents,3])
        for i in range(self.num_agents):
            #print(self.state[i],self.time[i],action[i])
            action_space = self.state_action_space(state[i])
            for j in range(3):
                if action_space[j][1] == self.state[i]:
                    mean_action[i,j] = count[action_space[j][2]]
                else:
                    mean_action[i,j] = count[action_space[j][2]]+1
        
        return self.state,time,reward, self.done, mean_action
    def mean_action(self,μ,epsilon,state,time):
        #μ is a n_agents*13*max_r dimention dictionary recording every observation's action
        #deterministic μ
        action = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            prob = self.rng.random()
            if prob <= epsilon:
                action[i] = np.random.randint(3)
            else:
                action[i] = μ[i,state[i]-1,time[i]]
        action = action.astype(int)
        state = np.copy(state)
        next_state,count = self.state_transition_func(state,action)
        mean_action = np.zeros([self.num_agents,3])
        for i in range(self.num_agents):
            #print(self.state[i],self.time[i],action[i])
            action_space = self.state_action_space(state[i])
            for j in range(3):
                if action_space[j][1] == next_state[i] :
                    mean_action[i,j] = count[action_space[j][2]]
                else:
                    mean_action[i,j] = count[action_space[j][2]]+1
            if (mean_action>=3).any():
                print(count,'error')
        return mean_action
    
        #stochastic μ
        '''action = np.zeros((self.num_agents,3))
        for i in range(self.num_agents):
            prob = self.rng.random()
            if prob <= epsilon:
                action[i] = np.ones(3)/3
            else:
                action[i] = μ[state[i]-1,time[i]]
        state = np.copy(state)
        _,count = self.state_transition_func_st(state,action)
        mean_action = np.zeros([self.num_agents,3])
        for i in range(self.num_agents):
            #print(self.state[i],self.time[i],action[i])
            action_space = self.state_action_space(state[i])
            for j in range(3):
                mean_action[i,j] = count[action_space[j][2]]+1 - action[i,j]

        return mean_action'''

    

    
    def state_transition_func_st(self, state, action):
        #action is a prob distribution 1*3
        next_state = np.copy(state)
        count = np.zeros(25)

        for j in range(len(state)):
            action_space = self.state_action_space(state[j])
            for i in range(len(action_space)):
                prob = action[j][i]
                next_state[j] = action_space[i][1]
                if state[j] == 1 and next_state[j] == 2:
                    count[0] += prob
                if state[j] == 2 and next_state[j] == 3:
                    count[1] +=prob
                if state[j] == 3 and next_state[j] == 4:
                    count[2] +=prob
                if state[j] == 4 and next_state[j] == 5:
                    count[3] +=prob
                if state[j] == 1 and next_state[j] == 6:
                    count[4] +=prob
                if state[j] == 1 and next_state[j] == 7:
                    count[5] +=prob
                if state[j] == 2 and next_state[j] == 7:
                    count[6] +=prob
                if state[j] == 2 and next_state[j] == 8:
                    count[7] +=prob
                if state[j] == 3 and next_state[j] == 9:
                    count[8] +=prob
                if state[j] == 4 and next_state[j] == 9:
                    count[9] +=prob
                if state[j] == 5 and next_state[j] == 9:
                    count[10] +=prob
                if state[j] == 6 and next_state[j] == 7:
                    count[11] +=prob
                if state[j] == 7 and next_state[j] == 8:
                    count[12] +=prob
                if state[j] == 8 and next_state[j] == 9:
                    count[13] +=prob
                if state[j] == 6 and next_state[j] == 10:
                    count[14] +=prob
                if state[j] == 7 and next_state[j] == 10:
                    count[15] +=prob
                if state[j] == 7 and next_state[j] == 11:
                    count[16] +=prob
                if state[j] == 8 and next_state[j] == 11:
                    count[17] +=prob
                if state[j] == 8 and next_state[j] == 12:
                    count[18] +=prob
                if state[j] == 9 and next_state[j] == 12:
                    count[19] +=prob
                if state[j] == 9 and next_state[j] == 13:
                    count[20] +=prob
                if state[j] == 10 and next_state[j] == 11:
                    count[21] +=prob
                if state[j] == 11 and next_state[j] == 12:
                    count[22] +=prob
                if state[j] == 12 and next_state[j] == 13:
                    count[23] +=prob

        next_state.astype(int)
        return next_state, count



