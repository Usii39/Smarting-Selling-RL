# 📈 SmartSellingRL：基於強化學習的量化交易智慧賣出系統

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Reinforcement Learning](https://img.shields.io/badge/Machine%20Learning-RL-orange.svg)

## 📖 專案簡介
本專案旨在解決量化交易中「會買不會賣」的痛點。有別於傳統的技術指標（如 MA、KD、RSI），本系統建構了自訂的強化學習環境，運用 **Q-Learning**、**SARSA** 與 **SARSA(λ)** 等演算法，讓 AI 代理人透過與歷史股價軌跡互動，動態尋找最佳的獲利了結時機點，並與傳統的 Buy & Hold 策略進行績效對比。

## ✨ 核心技術與亮點
為了符合業界軟體工程標準，本專案不僅止於研究用腳本，更具備以下工程化特色：
* **模組化與 OOP 設計**：將演算法、環境與技術指標封裝成獨立模組（`core/`, `utils/`），提高程式碼復用性與可維護性。
* **平行運算加速**：使用 `joblib` 進行多核心平行回測，大幅縮短動輒上萬回合的 RL 訓練與評估時間。
* **互動式數據儀表板**：結合 **Streamlit** 開發 Web App，讓使用者能透過 UI 介面即時調整學習率（Alpha）、折扣因子（Gamma）等超參數，並視覺化績效比較圖表。

## 📂 專案架構
```text
SmartSellingRL/
├── app.py                 # Streamlit 互動式 Web App 進入點
├── main.py                # 終端機大批量平行回測主程式
├── core/                  # 核心業務邏輯
│   ├── agent.py           # RL 演算法 (Q-learning, SARSA)
│   ├── environment.py     # RL 交易環境與狀態編碼
│   ├── backtest_engine.py # 績效評估與回測邏輯
│   └── pipeline.py        # 整合訓練與回測的流程控制
├── utils/                 # 輔助工具
│   ├── indicators.py      # 技術指標計算 (MA, KD, RSI, MACD)
│   └── psp2.py            # 模擬價格生成器 (fbm 等)
└── README.md              # 專案說明文件

<img width="1682" height="803" alt="rl_selling1" src="https://github.com/user-attachments/assets/d1b9a388-cc56-4986-b31e-c3419108768e" />
