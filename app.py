# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:57:59 2026

@author: 0125i
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, cpu_count

# --- 延續剛才的防呆機制，確保 Streamlit 找得到你的模組 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 引入你的回測大將
from core.pipeline import Big_backtest2

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="SmartSellingRL 量化交易系統", page_icon="📈", layout="wide")
st.title("📈 SmartSellingRL 強化學習交易策略展示")
st.markdown("透過 Q-Learning 與 SARSA 系列的演算法，動態尋找最佳賣出時機點。")

# --- 2. 側邊欄 (Sidebar)：放置互動參數 ---
st.sidebar.header("⚙️ 模型參數設定")

# 使用滑桿 (Slider) 讓使用者調整 RL 參數
alpha = st.sidebar.slider("學習率 (Alpha)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
gamma = st.sidebar.slider("折扣因子 (Gamma)", min_value=0.5, max_value=1.0, value=0.85, step=0.01)
epsilon = st.sidebar.slider("探索率 (Epsilon)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
lambd = st.sidebar.slider("資格跡 (Lambda)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
rf = st.sidebar.slider("銀行利率 (rf)", min_value=0.0, max_value=0.3, value=0.02, step=0.01)



st.sidebar.markdown("---")
st.sidebar.header("📊 環境與回測設定")
# Demo 為了講求速度，我們把迴圈次數 N 設小一點 (例如 5)，讓面試官點擊後幾秒內就能看到結果
demo_N = st.sidebar.number_input("平行測試次數 (N)", min_value=1, max_value=1000, value=10)
before = st.sidebar.number_input("觀察天數 (before)", min_value=1, max_value=10, value=3)
e_num_block = st.sidebar.number_input("單日報酬切點數", min_value=2, max_value=10, value=5)
M_num_block = st.sidebar.number_input("累積報酬切點數", min_value=2, max_value=10, value=9)
num_state = e_num_block**before*M_num_block
st.sidebar.info(f"🔢 狀態空間總數 (num_state)： **{num_state}**")
n_episodes = st.sidebar.number_input("回合數(episodes)", min_value=100, max_value=int(1e5), value=10*num_state,step=100)


train_len = st.sidebar.number_input("訓練窗格天數", min_value=30, max_value=1000, value=360,step=10)
test_len = st.sidebar.number_input("測試窗格天數", min_value=30, max_value=1000, value=120,step=10)

# # 固定參數
# train_len = 700
# test_len = 100


root_dir = r"D:\MS113_LiuDL\論文\Project\台股"
# 加上快取，避免每次操作網頁都重新讀取硬碟目錄
@st.cache_data
def get_stock_list(path):
    return os.listdir(path) if os.path.exists(path) else []

stock_list = get_stock_list(root_dir)

# --- 3. 執行按鈕與主畫面邏輯 ---
if st.sidebar.button("🚀 執行回測"):
    if not stock_list:
        st.error(f"找不到資料目錄：{root_dir}，請確認路徑是否正確。")
    else:
        # --- 替換原本的 st.spinner 區塊 ---
        
        # 1. 建立 Streamlit 畫面上的文字與進度條佔位符 (Placeholder)
        progress_text = st.empty()  # 用來顯示「目前進度：3 / 10」
        progress_bar = st.progress(0.0) # 產生一個從 0 開始的進度條

        progress_text.text(f"🚀 啟動平行運算引擎... (0/{demo_N})")

        # 2. 設定 Parallel，關鍵在於加入 return_as="generator"
        # 這樣 joblib 就不會等全部跑完才吐資料，而是跑完一個丟一個出來
        parallel_runner = Parallel(n_jobs=max(1, cpu_count()), return_as="generator")(
            delayed(Big_backtest2)(
                root_dir, stock_list, train_len, test_len, 
                before, e_num_block, M_num_block, 
                alpha, gamma, epsilon, lambd,
                rf, n_episodes
            ) for _ in range(demo_N)
        )

        # 3. 透過迴圈接收結果，並即時更新 UI
        reward_list = []
        for i, result in enumerate(parallel_runner):
            reward_list.append(result)
            
            # 計算目前的完成比例 (介於 0.0 到 1.0 之間)
            current_progress = (i + 1) / demo_N
            
            # 更新網頁上的進度條與文字
            progress_bar.progress(current_progress)
            progress_text.text(f"⏳ 模型訓練與平行回測執行中... ({i+1}/{demo_N})")
            
        # 4. 執行完畢後，可以選擇把進度條隱藏起來，讓畫面更乾淨
        progress_text.empty()
        progress_bar.empty()
        # ------------------------------------
        
        # 接著繼續原本的數據處理邏輯...
        my_columns = ['QL', 'SARSA', 'SARSA(𝜆)', 'B&H', '❓', 'MA', 'KD', 'RSI', 'MACD', '💰', 'QLf', 'SARSAf', 'SARSA(𝜆)f']
        reward_df = pd.DataFrame(np.vstack(reward_list), columns=my_columns)
    # ... 後面的 st.success 和圖表程式碼都不變
        rank_df = reward_df.rank(axis=1, ascending=False, method='min')

        st.success("回測完成！以下為本次實驗結果：")
        
        # 建立三個指標卡片
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("QL 中位數報酬", f"{reward_df['QL'].median():.2%}")
        with col2:
            st.metric("B&H (買進持有) 中位數報酬", f"{reward_df['B&H'].median():.2%}")
        with col3:
            st.metric("勝出差距", f"{reward_df['QL'].median() - reward_df['B&H'].median():.2%}")

        st.markdown("### 📊 各策略中位數報酬分佈比較 (%)")
        # Streamlit 原生支援簡單的圖表，也可以放 matplotlib/plotly 圖表
        # st.bar_chart(reward_df.applymap(lambda x: f"{x:.2%}").median())
        # sorted_median = (reward_df.median() * 100).sort_values(ascending=False)
        # st.bar_chart(sorted_median)

        st.bar_chart(100*reward_df.median().sort_values(ascending=False))

        st.markdown("### 📋 詳細統計數據")
        # 讓 DataFrame 在網頁上漂亮地展示
        st.dataframe(reward_df.describe().applymap(lambda x: f"{x:.2%}").T, width='stretch')

        st.markdown("### 📋 詳細排行數據")
        st.dataframe(rank_df.describe().T, width='stretch')


else:
    st.info("👈 請在左側設定參數，並點擊「執行回測」開始展示。")
    
    
    
    
    
    
    