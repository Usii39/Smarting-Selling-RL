# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:53:12 2026

@author: 0125i
"""

import numpy as np
from tqdm import tqdm
    
def q_learning(env, n_episodes=100, alpha=0.1, gamma=1.0, epsilon=0.1, seed=None, verbose=True):

   # TODO: initialization (same as SARSA)
   n_states = env.state_num
   n_actions = env.action_num
   Q = np.zeros((n_states, n_actions))
   # Q_hist = np.empty((0, n_states, n_actions))

   if seed is not None:
       np.random.seed(seed)

   def epsilon_greedy_policy(state):
        # TODO: epsilon-greedy selection
       if np.random.rand() < epsilon:
           return env.action_sample()
       else:
           return np.argmax(Q[state])

   if verbose:
       iter_tool =  tqdm(range(n_episodes),desc='Q-learning')
   else:
       iter_tool =  range(n_episodes)

   for idx in iter_tool:
       state = env.reset(idx)
       action = epsilon_greedy_policy(state)
       done = False

       while not done:
           next_state, reward, done = env.step(action)
           next_action = epsilon_greedy_policy(next_state)

           # TODO: Q-learning update rule
           Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

           # move forward
           state = next_state
           action = next_action

       # Q_hist = np.vstack([Q_hist, Q.reshape(1, n_states, n_actions)])
   # return Q_hist
   return Q 
   

def sarsa(env, n_episodes=100, alpha=0.1, gamma=1.0, epsilon=0.1, seed=None, verbose=True):

   # TODO: initialization (same as SARSA)
   n_states = env.state_num
   n_actions = env.action_num
   Q = np.zeros((n_states, n_actions))
   # Q_hist = np.empty((0, n_states, n_actions))

   if seed is not None:
       np.random.seed(seed)

   def epsilon_greedy_policy(state):
        # TODO: epsilon-greedy selection
       if np.random.rand() < epsilon:
           return env.action_sample()
       else:
           return np.argmax(Q[state])

   if verbose:
       iter_tool =  tqdm(range(n_episodes),desc='SARSA')
   else:
       iter_tool =  range(n_episodes)

   for idx in iter_tool:
       state = env.reset(idx)
       action = epsilon_greedy_policy(state)
       done = False

       while not done:
           next_state, reward, done = env.step(action)
           next_action = epsilon_greedy_policy(next_state)

           # TODO: SARSA update rule
           # Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
           Q[state, action] += alpha * (reward + gamma * Q[next_state,next_action] - Q[state, action])

           # move forward
           state = next_state
           action = next_action

       # Q_hist = np.vstack([Q_hist, Q.reshape(1, n_states, n_actions)])
   # return Q_hist
   return Q


def sarsa_lambda(env, n_episodes=100, alpha=0.1, gamma=1.0, epsilon=0.1, lambd=0.9, seed=None, verbose=True):
    """
    SARSA(lambda) 實作
    參數:
    - lambd: 資格跡衰減因子 (0~1)。
             0 = 普通 SARSA
             1 = 類似 Monte Carlo (直到結束才結算)
    """
    
    n_states = env.state_num
    n_actions = env.action_num
    
    # 1. 初始化 Q-table
    Q = np.zeros((n_states, n_actions))
    
    if seed is not None:
        np.random.seed(seed)

    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return env.action_sample()
        else:
            return np.argmax(Q[state])

    # 2. 開始訓練迴圈
    if verbose:
       iter_tool =  tqdm(range(n_episodes),desc='SARSA(𝜆)')
    else:
       iter_tool =  range(n_episodes)
    
    for idx in iter_tool:
        state = env.reset(idx)
        action = epsilon_greedy_policy(state)
        
        # [關鍵] 每個 Episode 開始時，資格跡 E 必須歸零
        E = np.zeros((n_states, n_actions))
        
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            
            # 計算 TD Error (delta)
            # SARSA 是 On-policy，所以這裡必須根據「實際選的下一步」來算
            if done:
                # 如果結束了，未來的價值就是 0
                delta = reward - Q[state, action]
            else:
                next_action = epsilon_greedy_policy(next_state)
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # [關鍵步驟 A] 增加當前狀態的資格跡 (Accumulating Trace)
            E[state, action] += 1
            
            # [關鍵步驟 B] 更新所有的 Q 值 (不僅僅是當前狀態)
            # Q(s,a) <- Q(s,a) + alpha * delta * E(s,a)
            # 這裡利用 numpy 的矩陣運算同時更新所有狀態，這是 SARSA(lambda) 的精髓
            Q += alpha * delta * E
            
            # [關鍵步驟 C] 衰減所有的資格跡
            # E(s,a) <- gamma * lambda * E(s,a)
            E *= gamma * lambd

            # 移動到下一步
            if not done:
                state = next_state
                action = next_action

    return Q