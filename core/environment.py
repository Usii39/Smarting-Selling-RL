# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:54:15 2026

@author: 0125i
"""

import numpy as np
from collections import deque


class SmartSelling_env:
    def __init__(self, sim_returns, before, e_num_block=5, M_num_block=10, rf=0.02, dt=1/252):
        self.sim_returns = sim_returns #log return
        self.before = before # 看最近幾期報酬e_{t-ell}
        self.rf = rf #銀行年利率
        self.dt = dt
      
        # self.bins_e = get_bins_quantile(sim_returns.flatten(), e_num_block) #單日報酬的切點
        self.bins_M = get_bins_quantile(np.cumsum(sim_returns,axis=1).flatten(), M_num_block)#累積報酬的切點

        self.bins_e = get_bins_std(sim_returns.flatten(), e_num_block) #單日報酬的切點
        # self.bins_M = get_bins_std(np.cumsum(sim_returns,axis=1).flatten(), M_num_block)#累積報酬的切點
        
        
        self.state_num = e_num_block**before * M_num_block
        self.action_num = 2#1:out, 0:hold
        

    def reset(self,idx):
        self.now_returns = deque(self.sim_returns[idx].flatten())
        self.e_t = deque([self.now_returns.popleft() for _ in range(self.before)],maxlen=self.before)
        self.M_t = np.sum(self.e_t)
        st = encoding(self.e_t, self.M_t, self.bins_e, self.bins_M)
        return st


    def step(self,action):
        self.e_t.append(self.now_returns.popleft())
        self.M_t += self.e_t[-1]
        next_st = encoding(self.e_t, self.M_t, self.bins_e, self.bins_M)

        
        if len(self.now_returns) <= 1:
            done = True
            reward = np.exp(self.M_t + self.now_returns.popleft()) -1#np.exp(self.M_t + self.now_returns[0]):明天賣掉拿到的錢
            
            
        elif action == 1:#out
            done = True
            reward = np.exp(self.M_t + self.now_returns.popleft()) * (1 + len(self.now_returns)*self.dt*self.rf) - 1
            
            
        elif action == 0:#hold:
            done = False
            reward = 0
            
        return next_st, reward, done
      
    def action_sample(self):
        return np.random.randint(0,2)
   

# --- 1. 建立切點 (Training階段做一次) ---
def get_bins_quantile(returns, n):
    """
    等量切分n個區塊
    returns: log(pi/p_{i-1}) or log(pi/p0)
    針對「單日報酬」e_t 或 「累積報酬」 M_t計算切點
    每一區數量相同
    """
    percentiles = np.linspace(0, 100, n + 1)
    values = np.percentile(returns, percentiles)
    values = values[1:-1] # 去掉 0% 和 100%
    values = sorted(list(set(values)))
    if len(values)+1 < n:
        print(f'區塊數量不足：{len(values)+1} < {n}')
    return values


def get_bins_std(returns, n, max_std=2.0):
    """
    等標準差切分n個區塊
    returns: log(pi/p_{i-1}) or log(pi/p0)
    針對「單日報酬」e_t 或 「累積報酬」 M_t計算切點
    每一區塊數量不同
    max_std :設定的最大標準差

    """
    std = np.std(returns)
    mu = np.mean(returns)
    
    percentiles = np.linspace(-max_std, max_std, n + 1)
    values = mu + std * percentiles
    values = values[1:-1] # 去掉 0% 和 100%
    values = sorted(list(set(values)))
    if len(values)+1 < n:
        print(f'區塊數量不足：{len(values)+1} < {n}')
    return values

# --- 2. 狀態編碼 (Step-by-step 執行) ---
def encoding(recent_returns, M_t, bins_return, bins_M):
    """
    將 (e_t序列, M_t) 混合編碼成唯一的 State ID
    
    參數:
    - recent_returns: list or array, 過去幾天的單日報酬 e.g., [0.01, -0.02, ...]
    - M_t: float, 目前的累積報酬
    - bins_return: list, 單日報酬的切點
    - bins_M: list, 累積報酬的切點
    """
    # 1. 處理近期的單日報酬 (Vector)
    recent_returns = np.array(recent_returns)
    # 找出每個 e_t 落在哪個區間 (0 ~ n_bins_ret-1)
    idx_returns = np.searchsorted(bins_return, recent_returns, side='right')
    
    # 2. 處理當前的長期報酬 (Scalar)
    # 找出 M_t 落在哪個區間 (0 ~ n_bins_M-1)
    idx_M = np.searchsorted(bins_M, [M_t], side='right')[0]
    
    # 3. 混合編碼 (Flattening)
    # 概念： ID = (e_t 的組合 ID) + (M_t 的 ID * e_t 組合的總可能性)
    
    # A. 先算 e_t 部分的權重
    n_bins_ret = len(bins_return) + 1  # 單日報酬有幾種狀態
    N = len(recent_returns)            # 看幾天
    weights_ret = n_bins_ret ** np.arange(N)
    
    id_from_returns = np.dot(idx_returns, weights_ret)
    
    # B. 計算 e_t 部分總共有多少種可能 (Offset)
    total_states_returns = n_bins_ret ** N
    
    # C. 加上 M_t 的貢獻
    # 最終 ID = (M_t區間編號 * 前面所有組合的數量) + 前面的組合ID
    final_index = (idx_M * total_states_returns) + id_from_returns
    
    return int(final_index)


