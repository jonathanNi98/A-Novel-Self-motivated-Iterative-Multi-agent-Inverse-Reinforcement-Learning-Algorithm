#!/usr/bin/python
from trainer import DQN
import argparse
import numpy as np
from utils import ReplayMemory
from env import OW
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main(args):
    env = OW(num_agents=args.N,max_r = args.max_r)
    num_actions = 3
    num_input = 13+1+3
    agent = DQN(num_input,num_actions,args.hidden_size,args.lr,args.batch_size)
    memory = ReplayMemory(args.total_steps,args.seed)

    #build a list for optimal policy
    #initial optimal policy
    step = 0
    n_episode = args.total_episodes
    μ = np.random.randint(3,size = (args.N,13,args.max_r))
    result = []
    for i_episode in range(n_episode):
        state,time = env.reset()
        for s_ in range(args.max_episode_length):
            a_ = env.mean_action(μ,args.eps,state,time)
            a_[:,2] = 0
            action,_ = agent.select_action(state,time,a_,args.eps)
            next_state,time_,reward,done,mean_action = env.step(action)
            next_a_ = env.mean_action(μ,args.eps,next_state,time_)
            next_a_[:,2] = 0
            for i in range(args.N):
                memory.push(state[i],action[i],reward[i],next_state[i],mean_action[i],time[i],time_[i],next_a_[i],i,done[i])
                step += 1
            state = next_state
            time = time_
            #print(state,action,time)
            if done.all() == 1:
                result.append(time.mean())
                if i_episode%200 == 0:
                    print(time.mean())
                break

        if step >= args.start_steps:
            μ = agent.train(args,memory,μ)
            if i_episode%100 == 0:
                agent.target_update()
                if args.eps >= 0.01:
                    args.eps = args.eps * 0.95
                agent.update_epsilon()
            #agent.optimal_pollicy()
    result = np.array(result) 
    x = moving_average(result,500)
    l1 = plt.plot(x)
    plt.show()
    '''memory = ReplayMemory(args.total_steps,args.seed)
    for i_episode in range(2000):
        state,time = env.reset_0()
        for s_ in range(args.max_episode_length):
            a_ = env.mean_action(μ,args.eps,state,time)
            action,_ = agent.select_action(state,time,a_,args.eps)
            next_state,time_,reward,done = env.step(action)
            next_a_ = env.mean_action(μ,args.eps,next_state,time_)
            for i in range(args.N):
                memory.push(state[i],action[i],reward[i],next_state[i],a_[i],time[i],time_[i],next_a_[i],done[i])
                step += 1
            state = next_state
            time = time_
            #print(state,action)
            if done.all() == 1:
                if i_episode%200 == 0:
                    print(time.mean())
                break'''
    


    
        
    '''
    

    #update Q_net
    for j in range(J):
        agent.train(memory)
        μ = agent.update_policy()'''
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy'))
    parser.add_argument('--env-name', default="OW_Net")
    parser.add_argument('--batch_size', type=int, help='Number of episodes per training batch',default = 64)
    parser.add_argument('-m', '--hidden_size', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 64)
    parser.add_argument('--target_update_interval', type=int, default=1,
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=100,
                    help='Steps sampling random actions (default: 100)')
    parser.add_argument('--max_episode_length', type=int, default=100,
                    help='max_episode_length (default: 10)')
    parser.add_argument('--total_episodes', type=int, default=40000, metavar='N',
                    help='Number of total episodes (default: 20000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.001)')
    parser.add_argument('--C', type=int, default=1, metavar='N',
                    help='number of categories')
    parser.add_argument('--N', type=int, default=2, metavar='N',
                    help='number of agents')
    parser.add_argument('--eps', type=float, default=0.99,
                    help='epsilon')
    parser.add_argument('-t', '--total_steps', type=int, help='Number of total time-steps',
                        default = int(10**3))
    parser.add_argument('-s', '--seed', type=int, help='random seed',
                        default = 1234)
    parser.add_argument('--update_frequence', type=int, help='Number of total time-steps',
                        default = int(1000))
    parser.add_argument('--max_r', type=int, help='Maximum reward',
                        default = int(1000))
    
    args = parser.parse_args()
    main(args)