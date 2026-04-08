# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:09:49 2026

@author: 0125i
"""

import sys
import os
# -強制把專案根目錄加進 Python 的系統環境變數 (sys.path) 裡
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
    
    
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# 只需要從你的 pipeline 叫出 Big_backtest 這個大將即可！
from core.pipeline import Big_backtest

def main():
    print("🚀 啟動 SmartSellingRL 系統...")
    
    # 1. 統一定義所有固定參數
    N = 32 * 1
    test_len = 100
    train_len = 700
    alpha = 0.1
    gamma = 0.85
    epsilon = 0.2
    lambd = 0.40
    rl_params = (alpha, gamma, epsilon, lambd)
    
    my_columns = ['QL', 'SARSA', 'SARSA(𝜆)', 'B&H', '❓', 'MA', 'KD', 'RSI', 'MACD', '💰', 'QLf', 'SARSAf', 'SARSA(𝜆)f']
    my_columns2 = ['QL', 'SARSA', 'SARSA(𝜆)', 'QLf', 'SARSAf', 'SARSA(𝜆)f']
    
    # 2. 尋找合適的狀態空間參數 (保留你原本的邏輯)
    params_set = set()
    best_list = []
    while True:
        before = 3
        e_num_block = 5
        M_num_block = 9
        if ((before, e_num_block, M_num_block) not in params_set) and (e_num_block**before * M_num_block < 1e4):
            params_set.add((before, e_num_block, M_num_block))
            break
        else:
            params_set.add((before, e_num_block, M_num_block))

    # 3. 設定資料路徑
    root_dir = r"D:\MS113_LiuDL\論文\Project\台股"
    stock_list = os.listdir(root_dir)
    
    print(f"📊 執行參數: 狀態數量={e_num_block**before*M_num_block}, RL參數={rl_params}")
    
    # 4. 執行平行運算 (呼叫 Big_backtest)
    reward_list = Parallel(n_jobs=cpu_count())(
        delayed(Big_backtest)(
            root_dir, stock_list, train_len, test_len, 
            before, e_num_block, M_num_block, 
            alpha, gamma, epsilon, lambd
        ) for i in tqdm(range(N), desc='平行回測進度')
    )
    # 5. 資料統整與分析
    reward_array = np.vstack(reward_list)
    reward_df = pd.DataFrame(reward_array, columns=my_columns)
    
    rank_df = reward_df.rank(axis=1, ascending=False, method='min')
    
    best_one = rank_df.describe().loc['mean', my_columns2].argmin()
    best_list.append(my_columns2[best_one])
    
    dev = rank_df.describe().at['mean', my_columns2[best_one]] - rank_df.describe().at['mean', 'B&H']
    dev2 = rank_df.describe().loc['mean', my_columns2].mean() - rank_df.describe().at['mean', 'B&H']

    # 6. 印出報告
    info0 = f'# {my_columns2[best_one]}： ({dev:.2f},{dev2:.2f})| {(before, e_num_block, M_num_block)} = {e_num_block**before*M_num_block}, ({rl_params}), len={train_len/test_len:.0f}'
    print('\n' + '='*50)
    print(info0)
    print('實際報酬比較')
    print(reward_df.describe().map(lambda x: f'{x:.2%}').to_string())
    print('\n報酬排名比較')
    print(rank_df.describe().map(lambda x: f'{x:.2f}').to_string())
    print('='*50)

# 這裡是 Python 專案的標準起手式，確保這個檔案是被「直接執行」而不是被「import」
if __name__ == '__main__':
    main()
    
    
    