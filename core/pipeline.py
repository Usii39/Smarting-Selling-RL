# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:51:46 2026

@author: 0125i
"""

import numpy as np
import pandas as pd
import os

from utils.indicators import compute_MA, compute_KD, compute_RSI, compute_MACD
from utils.psp2 import bm_simulation_single, fbm_simulation_single, hurst_dfa, hm_simulation, hm_params_final
from core.backtest_engine import backtest, backtest2
from core.environment import SmartSelling_env
from core.agents import q_learning, sarsa, sarsa_lambda


def Big_backtest(root_dir, stock_list, train_len, test_len, before=4, e_num_block=3, M_num_block=2, alpha=0.1, gamma=1.0, epsilon=0.1, lambd=0.3, rf=0.03, n_episodes=1000):
    steps = test_len
    # before = 5
    # e_num_block = 7
    # M_num_block = 2
    # times = 10
    # n_episodes = min(times * e_num_block**before*M_num_block,2e5)


    dt = 1/252

    
    safe = False
    while safe == False:
        idx = np.random.randint(0,len(stock_list))
        df = pd.read_parquet(os.path.join(root_dir,stock_list[idx]),columns=['close'])
        if len(df) > (train_len + test_len):
            safe = True

    train_idx = np.random.randint(0,len(df)-train_len-test_len)
    train_df = df.iloc[train_idx:train_idx+train_len]
    test_df = df.iloc[train_idx+train_len-1:train_idx+train_len+test_len]
    
    close = df['close']
    df['MA'] = compute_MA(close)
    df['K'], df['D'] = compute_KD(close)
    df['KD'] = df['K'] - df['D']
    df['RSI'] = compute_RSI(close)
    _, _, df['MACD_hist'] = compute_MACD(close)
    
    train_returns =  np.log(train_df).diff().dropna().to_numpy().flatten()
    test_returns = np.log(test_df).diff().dropna().to_numpy().flatten()

    mu = train_returns.mean().item()
    sigma = train_returns.std().item()
    hurst = hurst_dfa(train_returns)
    # hurst = 0.8

    sim_prices = bm_simulation_single(mu,sigma,state_n=n_episodes,steps=steps,seed=None,keep_one=True)
    sim_returns = np.diff(np.log(sim_prices),axis=1)

    sim_prices2 = fbm_simulation_single(mu,sigma,hurst,state_n=n_episodes,steps=steps,seed=None,keep_one=True)
    sim_returns2 = np.diff(np.log(sim_prices2),axis=1)

    # RL
    env = SmartSelling_env(sim_returns, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq = backtest(test_returns, best_action1, env, before, dt, rf)

    rs = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl = backtest(test_returns, best_action3, env, before, dt, rf)

    action_BH = np.zeros((test_len,))
    rbh = backtest2(test_returns, action_BH)

    action_random = np.zeros((test_len,))
    action_random[np.random.randint(0,test_len)] = 1
    rr = backtest2(test_returns, action_random)

    action_ma = df['MA'].iloc[-test_len:].to_numpy()
    action_ma = (action_ma < 0).astype(int)
    rma = backtest2(test_returns, action_ma, ta=True)

    action_kd = df['KD'].iloc[-test_len:].to_numpy()
    action_kd = (action_kd < 0).astype(int)
    rkd = backtest2(test_returns, action_kd, ta=True)

    action_rsi = df['RSI'].iloc[-test_len:].to_numpy()
    action_rsi  = (action_rsi > 70).astype(int)
    rrsi = backtest2(test_returns, action_rsi, ta=True)

    action_macd = df['MACD_hist'].iloc[-test_len:].to_numpy()
    action_macd  = (action_macd < 0).astype(int)
    rmacd = backtest2(test_returns, action_macd, ta=True)

    action_rf = np.zeros((test_len,))
    action_rf[0] = 1
    rrf = backtest2(test_returns, action_rf)

    
    # RL+fbm
    env = SmartSelling_env(sim_returns2, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq2 = backtest(test_returns, best_action1, env, before, dt, rf)

    rs2 = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl2 = backtest(test_returns, best_action3, env, before, dt, rf)
    
    return rq, rs, rsl, rbh, rr, rma, rkd, rrsi, rmacd, rrf, rq2, rs2, rsl2


#更強的技術分析買賣規則
def Big_backtest2(root_dir, stock_list, train_len, test_len, before=4, e_num_block=3, M_num_block=2, alpha=0.1, gamma=1.0, epsilon=0.1, lambd=0.3, rf=0.03, n_episodes=1000):
    steps = test_len
    # before = 5
    # e_num_block = 7
    # M_num_block = 2
    # times = 10
    # n_episodes = min(times * e_num_block**before*M_num_block,2e5)


    dt = 1/252

    
    safe = False
    while safe == False:
        idx = np.random.randint(0,len(stock_list))
        df = pd.read_parquet(os.path.join(root_dir,stock_list[idx]),columns=['close'])
        if len(df) > (train_len + test_len):
            safe = True

    train_idx = np.random.randint(0,len(df)-train_len-test_len)
    train_df = df.iloc[train_idx:train_idx+train_len]
    test_df = df.iloc[train_idx+train_len-1:train_idx+train_len+test_len]
    
    close = df['close']
    df['MA'] = compute_MA(close)
    df['K'], df['D'] = compute_KD(close)
    df['KD'] = df['K'] - df['D']
    df['RSI'] = compute_RSI(close)
    _, _, df['MACD_hist'] = compute_MACD(close)
    
    train_returns =  np.log(train_df).diff().dropna().to_numpy().flatten()
    test_returns = np.log(test_df).diff().dropna().to_numpy().flatten()

    mu = train_returns.mean().item()
    sigma = train_returns.std().item()
    hurst = hurst_dfa(train_returns)
    # hurst = 0.8

    sim_prices = bm_simulation_single(mu,sigma,state_n=n_episodes,steps=steps,seed=None,keep_one=True)
    sim_returns = np.diff(np.log(sim_prices),axis=1)

    fbm_times = 5
    sim_prices2 = fbm_simulation_single(mu/fbm_times,sigma/np.sqrt(fbm_times),hurst,state_n=n_episodes,steps=steps*fbm_times,seed=None,keep_one=True)
    sim_prices2 = sim_prices2[:,range(fbm_times-1,steps*fbm_times,fbm_times)]
    sim_returns2 = np.diff(np.log(sim_prices2),axis=1)

    # RL
    env = SmartSelling_env(sim_returns, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq = backtest(test_returns, best_action1, env, before, dt, rf)

    rs = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl = backtest(test_returns, best_action3, env, before, dt, rf)

    action_BH = np.zeros((test_len,))
    rbh = backtest2(test_returns, action_BH)

    action_random = np.zeros((test_len,))
    action_random[np.random.randint(0,test_len)] = 1
    rr = backtest2(test_returns, action_random)

    # TA
    # ... (前面的資料讀取部分保持不變) ...
    # ----------- 修改開始：優化 TA 賣出訊號 -----------

    # 1. MA (均線): ma5 由上到下穿越 ma20才賣
    ma_vals = df['MA'].iloc[-test_len:].to_numpy()
    ma_prev = df['MA'].shift(1).iloc[-test_len:].to_numpy()
    action_ma = ((ma_prev > 0) & (ma_vals <= 0)).astype(int)
    rma = backtest2(test_returns, action_ma, ta=True)
    
    # 2. KD (隨機指標): 優化為「高檔(>80)死亡交叉」
    # 邏輯：只有在 K > 80 的強勢區轉弱才賣，過濾掉盤整區的亂賣訊號
    k_vals = df['K'].iloc[-test_len:].to_numpy()
    kd_vals = df['KD'].iloc[-test_len:].to_numpy() # K - D
    # 條件：K < D (死叉) AND K > 80 (高檔)
    action_kd = ((kd_vals < 0) & (k_vals > 80)).astype(int)
    rkd = backtest2(test_returns, action_kd, ta=True)

    # 3. RSI (相對強弱): 優化為「跌破 70 賣出」
    # 邏輯：昨天 RSI > 70 (超買), 今天 RSI <= 70 (轉弱)
    rsi_vals = df['RSI'].iloc[-test_len:].to_numpy()
    rsi_prev = df['RSI'].shift(1).iloc[-test_len:].to_numpy() # 昨天的 RSI
    # 條件：昨天超買 且 今天跌破
    action_rsi = ((rsi_prev > 70) & (rsi_vals <= 70)).astype(int)
    rrsi = backtest2(test_returns, action_rsi, ta=True)

    # 4. MACD: 優化為「柱狀圖由正轉負」 (零軸下穿)
    # 邏輯：抓趨勢改變的第一個瞬間，而不是一直 < 0 就一直想賣
    macd_hist = df['MACD_hist'].iloc[-test_len:].to_numpy()
    macd_hist_prev = df['MACD_hist'].shift(1).iloc[-test_len:].to_numpy()
    # 條件：昨天紅柱(>0) 且 今天綠柱(<0)
    action_macd = ((macd_hist_prev > 0) & (macd_hist <= 0)).astype(int)
    rmacd = backtest2(test_returns, action_macd, ta=True)
    # ----------- 修改結束 -----------
    action_rf = np.zeros((test_len,))
    action_rf[0] = 1
    rrf = backtest2(test_returns, action_rf)

    
    # RL+fbm
    env = SmartSelling_env(sim_returns2, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq2 = backtest(test_returns, best_action1, env, before, dt, rf)

    rs2 = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl2 = backtest(test_returns, best_action3, env, before, dt, rf)
    
    return rq, rs, rsl, rbh, rr, rma, rkd, rrsi, rmacd, rrf, rq2, rs2, rsl2


#包含Heston
def Big_backtest3(root_dir, stock_list, train_len, test_len, before=4, e_num_block=3, M_num_block=2, alpha=0.1, gamma=1.0, epsilon=0.1, lambd=0.3, rf=0.03):
    steps = test_len
    # before = 5
    # e_num_block = 7
    # M_num_block = 2
    times = 10
    n_episodes = min(times * e_num_block**before*M_num_block,2e5)


    dt = 1/252

    
    safe = False
    while safe == False:
        idx = np.random.randint(0,len(stock_list))
        df = pd.read_parquet(os.path.join(root_dir,stock_list[idx]),columns=['close'])
        if len(df) > (train_len + test_len):
            safe = True

    train_idx = np.random.randint(0,len(df)-train_len-test_len)
    train_df = df.iloc[train_idx:train_idx+train_len]
    test_df = df.iloc[train_idx+train_len-1:train_idx+train_len+test_len]
    
    close = df['close']
    df['MA'] = compute_MA(close)
    df['K'], df['D'] = compute_KD(close)
    df['KD'] = df['K'] - df['D']
    df['RSI'] = compute_RSI(close)
    _, _, df['MACD_hist'] = compute_MACD(close)
    
    train_returns =  np.log(train_df).diff().dropna().to_numpy().flatten()
    test_returns = np.log(test_df).diff().dropna().to_numpy().flatten()

    mu = train_returns.mean().item()
    sigma = train_returns.std().item()
    hurst = hurst_dfa(train_returns)
    # hurst = 0.8

    sim_prices = bm_simulation_single(mu,sigma,state_n=n_episodes,steps=steps,seed=None,keep_one=True)
    sim_returns = np.diff(np.log(sim_prices),axis=1)

    sim_prices2 = fbm_simulation_single(mu,sigma,hurst,state_n=n_episodes,steps=steps,seed=None,keep_one=True)
    sim_returns2 = np.diff(np.log(sim_prices2),axis=1)


    mu_h, kappa, gamma, vv, rho = hm_params_final(train_returns, dt=1/252, n_particles=888, n_iter=220, burn_in=120, prop_std=None, verbose=False)
    _, _, sim_returns3 = hm_simulation(mu_h, kappa, gamma, vv, rho, dt, N=n_episodes, steps=steps, keep_one=False)
    
    
    # RL
    env = SmartSelling_env(sim_returns, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq = backtest(test_returns, best_action1, env, before, dt, rf)

    rs = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl = backtest(test_returns, best_action3, env, before, dt, rf)

    action_BH = np.zeros((test_len,))
    rbh = backtest2(test_returns, action_BH)

    action_random = np.zeros((test_len,))
    action_random[np.random.randint(0,test_len)] = 1
    rr = backtest2(test_returns, action_random)

    # TA
    # ... (前面的資料讀取部分保持不變) ...
    # ----------- 修改開始：優化 TA 賣出訊號 -----------

    # 1. MA (均線): ma5 由上到下穿越 ma20才賣
    ma_vals = df['MA'].iloc[-test_len:].to_numpy()
    ma_prev = df['MA'].shift(1).iloc[-test_len:].to_numpy()
    action_ma = ((ma_prev > 0) & (ma_vals <= 0)).astype(int)
    rma = backtest2(test_returns, action_ma, ta=True)
    
    # 2. KD (隨機指標): 優化為「高檔(>80)死亡交叉」
    # 邏輯：只有在 K > 80 的強勢區轉弱才賣，過濾掉盤整區的亂賣訊號
    k_vals = df['K'].iloc[-test_len:].to_numpy()
    kd_vals = df['KD'].iloc[-test_len:].to_numpy() # K - D
    # 條件：K < D (死叉) AND K > 80 (高檔)
    action_kd = ((kd_vals < 0) & (k_vals > 80)).astype(int)
    rkd = backtest2(test_returns, action_kd, ta=True)

    # 3. RSI (相對強弱): 優化為「跌破 70 賣出」
    # 邏輯：昨天 RSI > 70 (超買), 今天 RSI <= 70 (轉弱)
    rsi_vals = df['RSI'].iloc[-test_len:].to_numpy()
    rsi_prev = df['RSI'].shift(1).iloc[-test_len:].to_numpy() # 昨天的 RSI
    # 條件：昨天超買 且 今天跌破
    action_rsi = ((rsi_prev > 70) & (rsi_vals <= 70)).astype(int)
    rrsi = backtest2(test_returns, action_rsi, ta=True)

    # 4. MACD: 優化為「柱狀圖由正轉負」 (零軸下穿)
    # 邏輯：抓趨勢改變的第一個瞬間，而不是一直 < 0 就一直想賣
    macd_hist = df['MACD_hist'].iloc[-test_len:].to_numpy()
    macd_hist_prev = df['MACD_hist'].shift(1).iloc[-test_len:].to_numpy()
    # 條件：昨天紅柱(>0) 且 今天綠柱(<0)
    action_macd = ((macd_hist_prev > 0) & (macd_hist <= 0)).astype(int)
    rmacd = backtest2(test_returns, action_macd, ta=True)
    # ----------- 修改結束 -----------
    action_rf = np.zeros((test_len,))
    action_rf[0] = 1
    rrf = backtest2(test_returns, action_rf)

    
    # RL+fbm
    env = SmartSelling_env(sim_returns2, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq2 = backtest(test_returns, best_action1, env, before, dt, rf)

    rs2 = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl2 = backtest(test_returns, best_action3, env, before, dt, rf)


    # RL+heston
    env = SmartSelling_env(sim_returns3, before, e_num_block, M_num_block)

    Q1 = q_learning(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action1 = np.argmax(Q1,axis=1)

    Q2 = sarsa(env, n_episodes, alpha, gamma, epsilon, verbose=False)
    best_action2 = np.argmax(Q2,axis=1)

    Q3 = sarsa_lambda(env, n_episodes, alpha, gamma, epsilon, lambd, verbose=False)
    best_action3 = np.argmax(Q3,axis=1)

    # reward計算
    rq3 = backtest(test_returns, best_action1, env, before, dt, rf)

    rs3 = backtest(test_returns, best_action2, env, before, dt, rf)

    rsl3 = backtest(test_returns, best_action3, env, before, dt, rf)
    
    
    return rq, rs, rsl, rbh, rr, rma, rkd, rrsi, rmacd, rrf, rq2, rs2, rsl2, rq3, rs3, rsl3