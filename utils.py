"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import sys, os
import shutil
import glob
import csv
import random
import math

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, action_mean,time,next_time,next_a_,agent_n,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, action_mean,time,next_time,next_a_,agent_n,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, action_mean,time,next_time,next_a_,agent_n, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, action_mean,time,next_time,next_a_,agent_n, done

    def push_prob(self, state, action, action_prob, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, action_prob, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample_prob(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, action_prob, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, action_prob, reward, next_state, done

    def push_queue(self, si, sr, action, reward, next_si, next_sr, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (si, sr, action, reward, next_si, next_sr, done)
        self.position = (self.position + 1) % self.capacity

    def sample_queue(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        si, sr, action, reward, next_si, next_sr, done = map(np.stack, zip(*batch))
        return si, sr, action, reward, next_si, next_sr, done
        
    def push_two(self, si, sp, sr, action, reward, next_si, next_sp, next_sr, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (si, sp, sr, action, reward, next_si, next_sp, next_sr, done)
        self.position = (self.position + 1) % self.capacity

    def sample_two(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        si, sp, sr, action, reward, next_si, next_sp, next_sr, done = map(np.stack, zip(*batch))
        return si, sp, sr, action, reward, next_si, next_sp, next_sr, done

    def sample_trans(self,curr_sr,sample_action,next_srr):
        step = 0
        t = 0
        s = 0
        while s <= 300:
            batch = random.sample(self.buffer,1)
            si, sr, action, reward, next_si, next_sr, done = map(np.stack, zip(*batch))
            if (sr == curr_sr).all() and action == sample_action:
                s += 1
                if (next_sr == next_srr).all():
                    t += 1
            if step >= 1000:
                if s == 0:
                    return 0
                return t/s
                
            step += 1
        return t/s

