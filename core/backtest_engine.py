# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:45:05 2026

@author: 0125i
"""
import numpy as np
from collections import deque
# 如果你有用到 encoding，還要從你自己的檔案 import：
from core.environment import encoding

def backtest(test_returns, best_action, env, before, dt=1/252, rf=0.02):
    # for RL
    now_returns = deque(test_returns)
    e_t = deque([now_returns.popleft() for _ in range(before)],maxlen=before)
    M_t = np.sum(e_t)

    while True:
        st = encoding(e_t, M_t, env.bins_e, env.bins_M)
        action = best_action[st]
        
        if action == 1 or len(now_returns) == 0:
            if len(now_returns) > 0:
                next_r = now_returns.popleft()
                M_t += next_r
                reward = np.exp(M_t) * (1 + len(now_returns) * dt * rf) - 1
            else:
                reward = np.exp(M_t) - 1
            break
            
        else:
            next_r = now_returns.popleft()
            e_t.append(next_r)
            M_t += next_r
            
    return reward


def backtest2(test_returns, sell_action, dt=1/252,rf=0.02, ta=False):
    # for TA
    if sum(sell_action) == 0:
        sell_idx = len(test_returns)-1
    else:
        sell_idx = np.argmax(sell_action)
        if ta:
            sell_idx += 1
    reward = np.exp(np.sum(test_returns[:sell_idx+1])) * (1 + (len(test_returns)-sell_idx)*dt*rf) - 1
    return reward