#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:22:16 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class PavlovTiming(Dataset):
    def __init__(self, ncues=4, ntrials_per_cue=50, include_reward=True):
        self.include_reward = include_reward
        self.ncues = ncues
        self.reward_times = [10]
        self.reward_amounts = np.array([.1, .3, 1.2, 2.5, 5.0, 10.0, 20.0])
        self.reward_probabilities = np.array([0.06612594,0.09090909,0.14847358,0.15489467,0.31159175,0.1509519,0.07705306])
        self.reward_probabilities = self.reward_probabilities/self.reward_probabilities.sum()
        self.ntrials_per_cue = ntrials_per_cue
        self.ntrials = self.ncues * self.ntrials_per_cue
        self.make_trials()

    def make_trial(self, cue, iti):
        isi = self.reward_times[cue]
        trial = np.zeros((iti + isi + 2, self.ncues + 1))
        trial[iti, cue] = 1.0 # encode stimulus
        random_reward = np.random.choice(a=self.reward_amounts, size=None, replace=True, p=self.reward_probabilities)
        trial[iti + isi, -1] = random_reward #encode reward
        return trial

    def make_trials(self):
        cues = np.tile(np.arange(self.ncues), self.ntrials_per_cue)
        
        # ITI per trial
        ITIs = np.random.geometric(p=0.5, size=self.ntrials)
        
        # make trials
        self.trials = [self.make_trial(cue, iti) for cue, iti in zip(cues, ITIs)]
    
    def __getitem__(self, index):
        X = self.trials[index][:,:-1]       #cues and stimulus of the trial
        y = self.trials[index][:,-1:]       #reward of the trial
        
        # augment X with previous y
        if self.include_reward:
            X = np.hstack([X, y])           #practically the same as self.trial for that index

        return (torch.from_numpy(X), torch.from_numpy(y))   #returns tensor form of the array

    def __len__(self):
        return len(self.trials)
