#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:48:33 2022

@author: mobeets
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_loss(scores):
    plt.figure(figsize=(7,4))
    plt.plot(scores)
    plt.xlabel('# epochs')
    plt.ylabel('loss')

clr1 = np.array([198, 57, 255])/255     #purple
clr2 = np.array([0, 137, 64])/255       #green
clr3 = np.array([242, 90, 41])/255      #orange
clr4 = np.array([84, 170, 255])/255     #blue
COLORS = [clr1, clr2, clr3, clr4]

def plot_trials(trials):
    clrs = COLORS[:(trials[0].shape[1]-1)]
    clrs += ['k']
    
    plt.figure(figsize=(2,2))
    ymax = 0.8
    for t in range(len(trials)):
        for c in range(trials[t].shape[1]):
            yinds = np.where(trials[t][:,c])[0]
            for y in yinds:
                plt.plot(y*np.ones(2), [-t, -t+ymax], '-', color=clrs[c])
        xmx = trials[t].shape[0]
        plt.plot([0, xmx], -t + np.zeros(2), 'k-', alpha=0.25)
    plt.yticks(ticks=[], labels=[])
    plt.xlabel('time $\\rightarrow$')
    plt.ylabel('trials $\\rightarrow$')
    plt.gca().spines.get('left').set_visible(False)

def plot_predictions(responses, key='value', gamma=1.0):
    clrs = COLORS[:responses[0]['X'].shape[1]]
    if key == 'value':
        discount = lambda rs_future: np.sum([r * gamma ** (tau+1) if ~np.isnan(r) else 0.0 for tau,r in enumerate(rs_future)])
    
    plt.figure(figsize=(5,3))
    ymax = 0.85 # for spacing in plots
    tstep = max([trial[key].max()-trial[key].min() for trial in responses]) # for spacing in plots
    for t, trial in enumerate(responses):
        X = trial['X']
        rs = trial['y']
        t_stim = trial['iti']-1
        if trial['isi'] is not None:
            t_rew = trial['isi'] + t_stim
        else:
            t_rew = None
        xs = np.arange(X.shape[0])
        y = trial[key]
        clr = clrs[trial['cue']]
        
        # plot zero line (black, semi-transparent)
        plt.plot(xs, -t*tstep + np.zeros(len(xs)), 'k-', alpha=0.25)
        
        # plot stimulus (blue, semi-transparent)
        plt.plot(t_stim*np.ones(2), [-t*tstep, -t*tstep + tstep*ymax], '-', color='b', label='stim', alpha=0.5)
        
        # plot reward (green, semi-transparent)
        if t_rew is not None:
            plt.plot(t_rew*np.ones(2), [-t*tstep, -t*tstep + tstep*ymax], '-', alpha=0.5, label='rew', color='green')

        # plot prediction (purple, dots-and-dashes)
        plt.plot(xs[:len(y)], 0.95*ymax*y - t*tstep, '.-', color=clr, label='pred')

        if key == 'value':
            # plot true value (black, dotted)
            R = np.array([r + discount(rs[(i+1):]) for i,r in enumerate(rs)])
            plt.plot(xs[:-1], tstep*ymax*R[1:] - t*tstep, 'k--', alpha=0.9)
    
    plt.xlabel('time $\\rightarrow$')
    plt.ylabel(key)
    plt.yticks()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)
    plt.show()

def plot_hidden_activity(responses, key='Z', align_offset=1):
    clrs = COLORS[:responses[0]['X'].shape[1]]
    msz = 5
    plt.figure(figsize=(9,4))
    for trial in responses:
        t_stim = trial['iti']
        if trial['isi'] is not None:
            t_rew = trial['isi'] + t_stim
        else:
            t_rew = None
        clr = clrs[trial['cue']]
        
        Z = trial[key]
        plt.plot(Z[t_stim,0], Z[t_stim,1], 's', color=clr, markersize=5)
        plt.plot(Z[:,0], Z[:,1], '.-', color=clr, markersize=msz, alpha=0.5)
        if t_rew is not None:
            plt.plot(Z[t_rew,0], Z[t_rew,1], '*', color=clr, markersize=5)
        # plt.plot(Z[trial.iti+1+align_offset,0], Z[trial.iti+1+align_offset,1], '*', markersize=6, color=h.get_color())
    plt.plot(0, 0, 'k+')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.gca().set_xticks([]);
    plt.gca().set_yticks([]);
    plt.axis('equal')
    plt.show()
