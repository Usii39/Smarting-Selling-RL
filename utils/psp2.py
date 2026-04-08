# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 20:20:20 2025

@author: 0125i
price stochastic process(PSP)第二版，將第一版的錯誤更正
"""

import numpy as np 
from numpy.linalg import cholesky as chol
import matplotlib.pyplot as plt
from fbm import FBM
from copulae import GaussianCopula
from copulae.core import pseudo_obs
from scipy.stats import norm,skew, kurtosis
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm
from scipy.special import ndtr  
import matplotlib.pyplot as plt


def hurst_rs(x):
    x = np.asarray(x)
    N = len(x)
    Y = np.cumsum(x - np.mean(x))
    R = np.max(Y) - np.min(Y)
    S = np.std(x)
    H = np.log(R / S) / np.log(N)
    H = max(0.05,min(H,0.95))
    return H


def hurst_dfa(x, scales=None):
    x = np.asarray(x)
    N = len(x)
    y = np.cumsum(x - np.mean(x))

    if scales is None:
        scales = np.floor(np.logspace(np.log10(10), np.log10(N/4), 20)).astype(int)

    F = []
    for s in scales:
        n = N // s
        rms = []
        for i in range(n):
            seg = y[i*s:(i+1)*s]
            t = np.arange(s)
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F.append(np.mean(rms))

    H, _ = np.polyfit(np.log(scales), np.log(F), 1)
    H = max(0.05,min(H,0.95))

    return H


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
    
def make_near_pd(A, eps=1e-8):
    A = np.asarray(A, dtype=float)
    # ① 移除 NaN / Inf（必要）
    if not np.all(np.isfinite(A)):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    # ② 強制對稱（非常重要）
    A = 0.5 * (A + A.T)
    # ③ 對角線強制為 1（相關矩陣）
    np.fill_diagonal(A, 1.0)
    # ④ 再做 eig
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # ⑤ 再對稱一次（數值安全）
    A_pd = 0.5 * (A_pd + A_pd.T)
    np.fill_diagonal(A_pd, 1.0)

    return A_pd

def make_pd_cholesky_safe(A, max_tries=10):
    A = 0.5 * (A + A.T)
    jitter = 1e-6
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(A)
            return A
        except np.linalg.LinAlgError:
            A = A + jitter * np.eye(A.shape[0])
            jitter *= 10
    raise np.linalg.LinAlgError("Cannot make matrix PD even with jitter")


def fbm_simulation(col_mean,cov_matrix,corr_matrix,H_list,state_n=5000,steps=252,cor_open=True,seed=None,keep_one=False):
    """
    GfBm
    col_mean: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    H: hurst parameter
    state_n: 模擬的市場狀況數量
    steps: 模擬的步數
    模擬價格形狀: state_n X steps
    設期初價格為1, keep_one=True顯示期初價格
    模擬價格形狀: state_n X D X steps
    """
    
    if seed != None:
        np.random.seed(seed)
        
        
    D = len(col_mean)
    mu2 = np.array(col_mean).reshape(-1,1)
    cov_matrix = np.array(cov_matrix)
    sigma2 = np.sqrt(np.diag(cov_matrix)).reshape((-1,1))
    if cor_open:
        if not is_positive_definite(corr_matrix):
            corr_matrix = make_near_pd(corr_matrix)
    
    
    # 步驟2: 生成獨立fBm路徑 (無相關性)
    # 儲存原始fBm路徑
    fbm_samples = []
    for i in range(D):
        hurst = H_list[i]
        fbm_gen = FBM(n=steps, hurst=hurst, length=steps, method='daviesharte')
        paths = np.array([np.diff(fbm_gen.fbm()) for _ in range(state_n)])#是路徑，約為np.cumsum(epsilon*np.sqrt(dt), axis=1)
        fbm_samples.append(paths)
    
    
    fbm_matrix = np.array(fbm_samples)  # 形狀 (simulation, N)
    fbm_matrix = np.transpose(fbm_matrix,axes=(1,2,0))# shape: (state_n, steps, D)
    
    if cor_open:
        adjusted_fbm = np.zeros_like(fbm_matrix)
    
        for n in range(fbm_matrix.shape[0]):
            # 步驟3: 轉換為均勻分布邊際 (使用經驗CDF)ECDF
            uniform_matrix = pseudo_obs(fbm_matrix[n])  # 使用Copulae內建函數
            
            # 步驟4: 建立高斯Copula並導入目標相關性
            cop = GaussianCopula(dim=D)
            cop[:] = corr_matrix # 直接設定相關矩陣（需確保cor是正定矩陣）
            
            
            # 步驟5: 生成相關性調整後的均勻分布
            copula_uniform = cop.random(steps)
            
            
            # 步驟6: 將均勻分布轉換回fBm路徑(調整fbm_matrix的順序成為adjusted_fbm，是擁有正確cor的fbm)
            for i in range(D):
                # 保序轉換 (維持邊際分布)
                rank = np.argsort(uniform_matrix[:, i])
                adjusted_fbm[n,:, i] = fbm_matrix[n,:, i][rank][np.argsort(np.argsort(copula_uniform[:, i]))]
    else:
        adjusted_fbm = fbm_matrix

    adjusted_fbm = np.transpose(adjusted_fbm,axes=(0,2,1))
    growth = np.exp(np.cumsum(mu2 + sigma2*adjusted_fbm,axis=2))
    if keep_one:
        new_col = np.ones((state_n,D,1))
        growth = np.concatenate((new_col, growth), axis=2)
        
    return growth


def fbm_simulation_end(col_mean,cov_matrix,corr_matrix,H_list,state_n=5000,T=1,cor_open=True,seed=None):
    """
    GfBm 直接模擬到最後一期
    col_mean: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    H: hurst parameter
    state_n: 模擬的市場狀況數量
    T: 模擬的步數(相當於steps)
    模擬價格形狀: state_n X steps
    設期初價格為1, keep_one=True顯示期初價格
    模擬價格形狀: state_n X D X steps
    """
    if seed != None:
        np.random.seed(seed)
        
    D = len(col_mean)
    cov_matrix = np.array(cov_matrix)
    sigma2 = np.sqrt(np.diag(cov_matrix)).reshape((-1,1))
    mu2 = np.array(col_mean).reshape(-1,1)
    if not is_positive_definite(corr_matrix):
        corr_matrix = make_near_pd(corr_matrix)
    
    
    
    # 步驟2: 生成獨立fBm路徑 (無相關性)
    
    # 儲存原始fBm路徑
    fbm_samples = []
    for i in range(D):
        hurst = H_list[i]
        fbm_gen = FBM(n=1, hurst=hurst, length=T, method='daviesharte')
        paths = np.array([fbm_gen.fbm()[-1] for _ in range(state_n)])
        fbm_samples.append(paths)
    
    fbm_matrix = np.array(fbm_samples).T  # 形狀 (simulation, N)
    
    if cor_open:
        
        # 步驟3: 轉換為均勻分布邊際 (使用經驗CDF)ECDF
        uniform_matrix = pseudo_obs(fbm_matrix)  # 使用Copulae內建函數
        
        # 步驟4: 建立高斯Copula並導入目標相關性
        cop = GaussianCopula(dim=D)
        cop[:] = corr_matrix # 直接設定相關矩陣（需確保cor是正定矩陣）
        
        
        # 步驟5: 生成相關性調整後的均勻分布
        copula_uniform = cop.random(state_n)
        
        
        # 步驟6: 將均勻分布轉換回fBm路徑(調整fbm_matrix的順序成為adjusted_fbm，是擁有正確cor的fbm)
        adjusted_fbm = np.zeros_like(fbm_matrix)
        for i in range(D):
            # 保序轉換 (維持邊際分布)
            rank = np.argsort(uniform_matrix[:, i])
            adjusted_fbm[:, i] = fbm_matrix[:, i][rank][np.argsort(np.argsort(copula_uniform[:, i]))]
        
    else:
        adjusted_fbm = fbm_matrix
        
    growth = np.exp(mu2*T + sigma2*adjusted_fbm.T)  # 模擬報酬倍率
    growth = growth.T
    return growth


def fbm_simulation_single(mu,sigma,H,state_n=5000,steps=252,seed=None,keep_one=False):
    """
    fGBM
    mu: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    sigma:  std of log return
    H: hurst parameter
    state_n: 模擬的市場狀況數量
    steps: 模擬的步數
    模擬價格形狀: state_n X steps
    設期初價格為1, keep_one=True顯示期初價格
    """

    if seed is not None:
        np.random.seed(seed)
        
    fbm_gen = FBM(n=steps, hurst=H, length=steps, method='daviesharte')
    fbm_increments = np.array([np.diff(fbm_gen.fbm()) for _ in range(state_n)])

    growth = np.exp(np.cumsum(mu + sigma * fbm_increments,axis=1))

    if keep_one:
        growth = np.hstack([np.ones((state_n, 1)), growth])

    return growth



    
def bm_simulation(col_mean,cov_matrix,corr_matrix,state_n=5000,steps=252,vrt='',cor_open=True,seed=None,keep_one=False):
    """
    GBM
    col_mean: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    state_n: 模擬的市場狀況數量
    steps: 模擬的步數
    T: 模擬的年限
    模擬價格形狀: state_n X D X steps
    
    vrt(variance reduction techniques):
        av: Antithetic Variates:只產生一半隨機變數， mean=0
        mm: Moment matching: 對隨機變數標準化~N(0,1) 
    """
    
    if seed != None:
        np.random.seed(seed)
    
    D = len(col_mean)
    cov_matrix = np.array(cov_matrix)
    sigma2 = np.sqrt(np.diag(cov_matrix)).reshape((-1,1))
    mu2 = np.array(col_mean).reshape(-1,1)
    if cor_open:
        if not is_positive_definite(corr_matrix):
            corr_matrix = make_near_pd(corr_matrix)
        
    
    if vrt == 'av':
        n30 = np.random.normal(size=(state_n//2,D,steps))
        # epsilon = np.vstack([n30,-n30])
        n3 = np.concatenate([n30,-n30],axis=0)
        # n3.mean(axis=0) #檢查=0    
        
    elif vrt == 'mm':
        n30 = np.random.normal(size=(state_n,D,steps))
        n30_mean = n30.mean(axis=0, keepdims=True)  # shape: (1, D, steps)
        n30_std = n30.std(axis=0, keepdims=True)    # shape: (1, D, steps)
        n3 = (n30 - n30_mean) / n30_std
        # n3.mean(axis=0)
        # n3.std(axis=0)
    else:
        n3 = np.random.normal(size=(state_n,D,steps))
        
    # epsilon = L_G2 @ n3

    if cor_open: 
        L_G2 = chol(corr_matrix)
        epsilon = np.einsum('ij,sjk->sik',L_G2,n3)
    else:
        epsilon = n3
        
        
    growth = np.exp(np.cumsum(mu2 + sigma2*epsilon,axis=2))
    if keep_one:
        new_col = np.ones((state_n,D,1))
        growth = np.concatenate((new_col, growth), axis=2)
        
    return growth

    
def bm_simulation_end(col_mean,cov_matrix,corr_matrix,state_n=5000,T=1,vrt='',cor_open=True,seed=None):
    """
    col_mean: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    GBM 直接模擬到最後一期
    state_n: 模擬的市場狀況數量
    T: 模擬多少個dt
    cor_open: True=相關性價格路徑，False=無相關性價格路徑
    """
    if seed != None:
        np.random.seed(seed)

    D = len(col_mean)
    cov_matrix = np.array(cov_matrix)
    sigma2 = np.sqrt(np.diag(cov_matrix)).reshape((-1,1))
    mu2 = np.array(col_mean).reshape(-1,1)
    if not is_positive_definite(corr_matrix):
        corr_matrix = make_near_pd(corr_matrix)
    
    
    
    if vrt == 'av':
        n30 = np.random.normal(scale=np.sqrt(T),size=(state_n//2,D))
        n3 = np.concatenate([n30,-n30],axis=0)
        
    elif vrt == 'mm':
        n30 = np.random.normal(scale=np.sqrt(T),size=(state_n,D))
        n30_mean = n30.mean(axis=0, keepdims=True)  # shape: (1, D, steps)
        n30_std = n30.std(axis=0, keepdims=True)    # shape: (1, D, steps)
        n3 = (n30 - n30_mean) / n30_std
    else:
        n3 = np.random.normal(scale=np.sqrt(T),size=(state_n,D))
    
    if cor_open: 
        L_G2 = chol(corr_matrix)
        epsilon = np.einsum('ij,jk->ik',L_G2,n3.T)
    else:
        epsilon = n3.T
        
    growth = np.exp(mu2*T + sigma2*epsilon)
    growth = growth.T

    return growth



def bm_simulation_single(mu,sigma,state_n=5000,steps=252,vrt='',seed=None,keep_one=False):
    """
    GBM
    mu: mean of log returns，dt和mu的時間尺度一樣。例如mu是用日資料算的，則每一步(step)都代表一天,dt=1天。
    sigma:  std of log return
    state_n: 模擬的市場狀況數量
    steps: 模擬的步數
    模擬價格形狀: state_n X steps
    設期初價格為1, keep_one=True顯示期初價格

    vrt(variance reduction techniques):
        av: Antithetic Variates:只產生一半隨機變數， mean=0
        mm: Moment matching: 對隨機變數標準化~N(0,1) 
    """
    if seed != None:
        np.random.seed(seed)
        
    if vrt == 'av':
        n30 = np.random.normal(size=(state_n//2,steps))
        # epsilon = np.vstack([n30,-n30])
        epsilon = np.concatenate([n30,-n30],axis=0)
        # n3.mean(axis=0) #檢查=0    
        
    elif vrt == 'mm':
        n30 = np.random.normal(size=(state_n,steps))
        n30_mean = n30.mean(axis=0, keepdims=True)  # shape: (1, D, steps)
        n30_std = n30.std(axis=0, keepdims=True)    # shape: (1, D, steps)
        epsilon = (n30 - n30_mean) / n30_std
        # n3.mean(axis=0)
        # n3.std(axis=0)
    else:
        epsilon = np.random.normal(size=(state_n,steps))
        
    
    growth = np.exp(np.cumsum(mu + sigma*epsilon,axis=1))
    if keep_one:
        growth = np.hstack([np.ones((state_n,1)),growth])
    return growth




def hm_simulation(mu_s, kappa_s, gamma_s, vv_s, rho_s, corr_matrix, dt=1/252, state_n=5000,steps=252,vrt='',seed=None,keep_one=False):
    """
    Heston Modle 多股票版本
    需要注意個別股票和自己的波動度相關係數以及股票間的報酬相關係數，需要符合條件。
    模型參數(年化)：
    股票list的參數組
    - mu: 漲幅率（可為0）
    - gamma: 變異數長期平均值
    - kappa: 均值回歸速度
    - vv: 波動率的波動度
    - rho: Brownian motion 的相關係數
    - N: 模擬路徑數
    - steps: 每條路徑的時間步數
    - corr_matrix: 股票報酬間的相關係數
    - dt: 一步是幾年，ex天是1/252
    X: log return
    
    """
    
    if seed != None:
        np.random.seed(seed)
    
    
    D = len(mu_s)
    
    mu2 = np.array(mu_s)
    kappa2 = np.array(kappa_s)
    gamma2 = np.array(gamma_s)
    vv2 = np.array(vv_s)
    
    rho_s = np.array(rho_s).flatten()
    rho_s = np.clip(rho_s, -0.95, 0.95)

    
    big_corr = np.block([[corr_matrix, np.diag(rho_s)],
                   [np.diag(rho_s), np.eye(D)]])


        
    if not is_positive_definite(big_corr):
        # big_corr = make_near_pd(big_corr)
        big_corr = make_pd_cholesky_safe(big_corr)

    
    L_G2 = chol(big_corr)


    n3 = np.random.normal(size=(state_n,D*2,steps))
    # epsilon = L_G2 @ n3
    epsilon = np.einsum('ij,sjk->sik',L_G2,n3)
    Wts = epsilon[:,:D,:] * np.sqrt(dt)
    Wtv = epsilon[:,D:,:] * np.sqrt(dt)


    Vt = np.zeros((state_n,D,steps+1))#年化變異數
    Xt = np.zeros((state_n,D,steps+1))# log returns
    Vt[...,0] = np.tile(gamma2.T,(state_n,1))
    
    for t in range(steps):
        Vt[...,t+1] = np.abs(Vt[...,t]+kappa2*(gamma2-Vt[...,t])*dt+vv2*np.sqrt(Vt[...,t])*Wtv[...,t])
        Xt[...,t+1] = (mu2 - Vt[...,t]/2)*dt + np.sqrt(Vt[...,t]) * Wts[...,t]
    
    growth = np.exp(np.cumsum(Xt,axis=2))
    
    
    if not keep_one:
        # 如果不想要 t=0 的初始點，全部切掉第一欄
        growth = growth[...,1:]
        Vt = Vt[..., 1:]
        Xt = Xt[..., 1:]
    
    return growth, Vt, Xt



def hm_simulation_single(mu, kappa, gamma, vv, rho, dt=1/252, state_n=5000, steps=252, seed=None, keep_one=False):
    """
    Heston Modle
    模型參數(年化)：
    - mu: 漲幅率（可為0）
    - gamma: 變異數長期平均值
    - kappa: 均值回歸速度
    - vv: 波動率的波動度
    - rho: Brownian motion 的相關係數
    - N: 模擬路徑數
    - steps: 每條路徑的時間步數
    - dt: 一步是幾年，ex天是1/252
    X: log return
    """
    if seed is not None:
        np.random.seed(seed)
        
    #check Feller condition
    # if 2*kappa*theta < Lambda**2:

    Z1 = np.random.normal(size=(state_n,steps))
    Z2 = np.random.normal(size=(state_n,steps))
    Wtv = Z1 * np.sqrt(dt)
    Wts = (rho * Z1 + np.sqrt(1-rho**2) * Z2) * np.sqrt(dt)

    Vt = np.zeros((state_n,steps+1))#年化變異數
    Xt = np.zeros((state_n,steps+1))# log returns
    
    
    Vt[:,0] = gamma
    for t in range(steps):
        Vt[:,t+1] = np.abs(Vt[:,t]+kappa*(gamma-Vt[:,t])*dt+vv*np.sqrt(Vt[:,t])*Wtv[:,t])
        Xt[:,t+1] = (mu - Vt[:,t]/2)*dt + np.sqrt(Vt[:,t]) * Wts[:,t]
    
    
    growth = np.exp(np.cumsum(Xt,axis=1))
    if not keep_one:
        # 如果不想要 t=0 的初始點，全部切掉第一欄
        growth = growth[:, 1:]
        Vt = Vt[:, 1:]
        Xt = Xt[:, 1:]
    
    return growth, Vt, Xt


       
def hm_params_from_rolling(x, h=50, delta=1/252):
    """
    使用 Scipy 進行 MLE 估計
    kappa, gamma, vv, mu, rho
    # """
    # vj = 1/delta*pd.Series(x).rolling(h,center=True).var().dropna()  
    # xbar = pd.Series(x).rolling(h,center=True).mean().dropna()

    vj = 1/delta*pd.Series(x).rolling(h,center=False).var().dropna()  
    xbar = pd.Series(x).rolling(h,center=False).mean().dropna()
    
    dvj = vj.diff().shift(-1).dropna()
    vj = vj.iloc[:-1]     
    xbar = xbar.iloc[:-1]
    
    x = x[vj.index]
    dvj = dvj.to_numpy()
    vj = vj.to_numpy()
    xbar = xbar.to_numpy()
    
    
    # 負對數概似函數
    def neg_log_likelihood1(params):
        kappa, gamma, v_v = params
        mu_j = kappa * (gamma - vj) * delta
        sigma_j = v_v * np.sqrt(vj * delta)
        z = (dvj - mu_j)/sigma_j
        
        log_likelihood = np.sum(
            -np.log(sigma_j)  - 1/2*z**2     
            )
        return -log_likelihood
    
    # 起始參數
    v_mean = np.mean(vj)
    v_std = np.std(vj)
    start_params = [1.0, v_mean, v_std]
    # start_params = [1.8, 0.1, 0.2]
    
    # 參數邊界（確保正數）
    bounds = [(0.5, 5), (1e-7, 1), (1e-7, 5)]
    # bounds = [(1e-7, 10), (1e-7, 1), (1e-7, 5)]
    # 優化
    result = minimize(neg_log_likelihood1, start_params, bounds=bounds, method='L-BFGS-B')
    # result = minimize(neg_log_likelihood, start_params, bounds=bounds, method='TNC')
    # result = minimize(neg_log_likelihood, start_params, bounds=bounds, method='SLSQP')
    # print(result.fun)
    kappa, gamma, v_v = result.x[0], result.x[1], result.x[2]
    
    # 負對數概似函數
    def neg_log_likelihood2(mu):
        
        mu_j = (mu-vj/2)*delta
        sigma_j = np.sqrt(vj*delta)
        z = (x - mu_j)/sigma_j
        
        
        log_likelihood = np.sum(
            -np.log(sigma_j)  - 1/2*z**2     
            )
        
        return -log_likelihood
    
    # 起始參數
    start_params = np.mean(x)/delta
    # print(start_params)
    
    # 優化
    result = minimize(neg_log_likelihood2, start_params, method='L-BFGS-B')
    mu = result.x[0]
    
    
    rho = np.corrcoef(xbar, vj)[0,1]
    # rho = np.corrcoef(x, dvj)[0,1]

    #調整rho
    # if abs(rho) > 0.1:
    #     print('adjust rho ')
    #     rho = max(-1, 3*rho)    
    
    return mu, kappa, gamma, v_v, rho



def hm_params_from_filtered(x, filtered_var, delta=1/252):
    """
    使用 Particle Filter 產生的變異數路徑進行 MLE 參數估計
    
    參數:
    - x: log returns (長度 N)
    - filtered_var: 濾波後的變異數路徑 (長度 N 或 N+1)
    - delta: 時間步長
    
    回傳:
    - mu, kappa, gamma, vv, rho
    """
    # 1. 數據對齊與前處理
    # 根據你的模擬邏輯: Vt[t] 驅動 Xt[t+1] 和 Vt[t+1]
    # 我們假設 x 和 filtered_var 的長度是一樣的，或者 x 是從 t=1 開始的 returns
    
    x = np.array(x)
    v = np.array(filtered_var)
    
    # 確保長度一致，取最小值
    n = min(len(x), len(v))
    x = x[:n]
    v = v[:n]
    
    # 切片邏輯 (Slicing):
    # v_curr: V_t (從 0 到 N-2)
    # v_next: V_{t+1} (從 1 到 N-1)
    # x_next: X_{t+1} (從 1 到 N-1)
    
    v_curr = v[:-1]      # V_0 ... V_{T-1}
    v_next = v[1:]       # V_1 ... V_T
    x_next = x[1:]       # X_1 ... X_T

    # 確保變異數為正 (避免 log 報錯)
    v_curr = np.maximum(v_curr, 1e-8)
    
    # 計算變異數的增量 dV
    dv = v_next - v_curr

    # --- 第一步: 估計 V 的動態 (Kappa, Gamma, Sigma) ---
    # V_{t+1} - V_t = kappa(gamma - V_t)dt + sigma*sqrt(V_t)*dW_v
    # 負對數概似函數
    def neg_log_likelihood_v(params):
        k, g, sig = params
        # 計算理論的 Drift 和 Diffusion
        drift = k * (g - v_curr) * delta
        diffusion = sig * np.sqrt(v_curr * delta)
        
        # 計算標準化殘差 (z score)
        z = (dv - drift) / diffusion
        
        # Log Likelihood (忽略常數項)
        # LL = -log(diffusion) - 0.5 * z^2
        ll = -np.sum(np.log(diffusion)) - 0.5 * np.sum(z**2)
        return -ll

    # 初始猜測與邊界
    # 這裡的初始值可以直接用數據的統計特徵
    v_mean = np.mean(v)
    v_std = np.std(v)
    start_v = [1.0, v_mean, v_std] 
    bounds_v = [(0.1, 10.0), (0.001, 1.0), (0.001, 2.0)]
    # bounds_v = [(0.5, 5.0), (1e-7, 1.0), (1e-7, 5.0)]

    
    res_v = minimize(neg_log_likelihood_v, start_v, bounds=bounds_v, method='L-BFGS-B')
    k_est, g_est, vv_est = res_v.x

    # --- 第二步: 估計 Mu ---
    # X_{t+1} = (mu - V_t/2)dt + sqrt(V_t)*dW_s
    
    def neg_log_likelihood_mu(mu_val):
        drift = (mu_val - v_curr / 2) * delta
        diffusion = np.sqrt(v_curr * delta)
        
        z = (x_next - drift) / diffusion
        ll = -np.sum(np.log(diffusion)) - 0.5 * np.sum(z**2)
        return -ll

    
    # 起始參數
    start_mu = np.mean(x)/delta
    # print(start_params)
    
    res_mu = minimize(neg_log_likelihood_mu, start_mu, method='L-BFGS-B')
    mu_est = res_mu.x[0]

    # --- 第三步: 估計 Rho ---
    # 這是最精華的部分：直接計算殘差的相關性
    # rho 的定義是兩個布朗運動增量 dW_t^v 和 dW_t^s 之間的相關係數
    # 1. 變異數的殘差residual (代表 dW_v)
    expected_dv = k_est * (g_est - v_curr) * delta
    std_dv = vv_est * np.sqrt(v_curr * delta)
    residuals_v = (dv - expected_dv) / std_dv
    
    # 2. 回報的殘差 (代表 dW_s)
    expected_dx = (mu_est - v_curr / 2) * delta
    std_dx = np.sqrt(v_curr * delta)
    residuals_x = (x_next - expected_dx) / std_dx
    
    # 3. 計算相關係數
    rho_est = np.corrcoef(residuals_x, residuals_v)[0, 1]

    return mu_est, k_est, g_est, vv_est, rho_est




def hm_particle_filter(returns, params, N=3000, dt=1/252):
    """
    Algorithm 3.1: Particle Filtering of the Volatility Process
    
    參數:
    - returns: 觀察到的 log returns (x_1, ..., x_T)
    - params: (kappa, gamma, vv, mu, rho)
    - N: 粒子數量 (Particles)
    - dt: 時間步長
    
    回傳:
    - filtered_vol: 濾波後的波動率路徑 E[V_t | x_{1:t}]
    - log_likelihood: 整個模型的 Log Likelihood (Eq 3.23)
    """
    returns = np.asarray(returns)
    mu, kappa, gamma, vv, rho = params
    
    # --- 防呆檢查 (Debug Print) ---
    # print("--- 濾波器參數檢查 ---")
    # print(f"Mu (漂移項): {mu:.4f}")
    # print(f"Kappa (回歸速度): {kappa:.4f}")
    # print(f"Gamma (長期均值): {gamma:.4f}")
    # print(f"Sigma (波動度):   {vv:.4f}")
    # print(f"Rho   (相關係數): {rho:.4f}")
    # print("----------------------")
    
    T_steps = len(returns)
    
    # 1. 初始化粒子
    # Mean = shape * scale = (2kg/s^2) * (s^2/2k) = gamma
    particles_v = np.random.gamma(shape=(2.0 * kappa * gamma / vv**2), 
                                  scale=(vv**2 / (2.0 * kappa)), 
                                  size=N)
    
    filtered_var = np.zeros(T_steps)
    total_log_likelihood = 0.0
    weights = np.ones(N) / N
    
    for t in range(T_steps):
        x_obs = returns[t]
        
        # A. 預測 (Prediction)
        z_v = np.random.normal(0, 1, N)
        w_v = z_v * np.sqrt(dt) # 這是 Brownian Motion 的增量 dW
        
        v_prev = particles_v
        v_prev_pos = np.abs(v_prev)
        
        # Euler Discretization for V
        particles_v_new = v_prev_pos + kappa * (gamma - v_prev_pos) * dt + \
                          vv * np.sqrt(v_prev_pos) * w_v
        particles_v_new = np.abs(particles_v_new)
        
        # B. 校正 (Correction)
        # 計算 x_j 的條件分佈參數
        # E[x_j] = drift + rho * sqrt(V) * w_v
        cond_mean = (mu - v_prev_pos / 2) * dt + rho * np.sqrt(v_prev_pos) * w_v
        # Std[x_j] = sqrt((1-rho^2) * V * dt)
        cond_std = np.sqrt((1 - rho**2) * v_prev_pos * dt)
        
        # 計算 Log PDF
        # 加上極小值 1e-10 防止除以零
        log_probs = -0.5 * np.log(2 * np.pi) - np.log(cond_std + 1e-10) - \
                    0.5 * ((x_obs - cond_mean) / (cond_std + 1e-10))**2
        
        # Log-Sum-Exp Trick 更新 Likelihood
        max_log_prob = np.max(log_probs)
        likelihood_t = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob))) - np.log(N)
        total_log_likelihood += likelihood_t
        
        # 更新權重
        log_weights_new = log_probs
        max_log_w = np.max(log_weights_new)
        
        weights_unnormalized = np.exp(log_weights_new - max_log_w)
        
        # 防呆
        w_sum = np.sum(weights_unnormalized)
        
        if (not np.isfinite(w_sum)) or (w_sum <= 0):
            weights = np.ones(N) / N
        else:
            weights = weights_unnormalized / w_sum
        
        # C. 估計 (Estimation) - 加權平均
        filtered_var[t] = np.sum(particles_v_new * weights)
        
        # D. 重抽樣 (Resampling)
        indices = np.random.choice(np.arange(N), size=N, p=weights)
        particles_v = particles_v_new[indices]
        weights = np.ones(N) / N

    return filtered_var, total_log_likelihood



# 1. 定義先驗分佈 (Prior)
# 我們使用均勻分佈 (Uniform Prior) 作為邊界限制
# 如果參數超出合理範圍，回傳 -inf (代表機率為 0)
def get_log_prior(params):
    mu, kappa, gamma, vv, rho = params
    
    # 設定參數的合理邊界 (根據 Heston 模型特性)
    if not (-1.0 < mu < 1.0): return -np.inf
    if not (0.01 < kappa < 10.0): return -np.inf      # kappa 必須為正
    if not (0.001 < gamma < 1.0): return -np.inf     # gamma (長期變異數) 合理範圍
    if not (0.01 < vv < 2.0): return -np.inf         # vol of vol
    if not (-0.99 < rho < 0.99): return -np.inf      # rho 在 -1 到 1 之間
    
    # 滿足 Feller Condition (選用，太嚴格可以先拿掉)
    if 2 * kappa * gamma < vv**2: return -np.inf 

    return 0.0 # 在範圍內，log(1) = 0 (Uniform Prior)


def run_pmcmc(returns, initial_params, n_iter=2000, n_particles=500, proposal_std=None, dt=1/252, verbose=True):
    """
    PMCMC 主程式
    
    參數:
    - returns: 觀測到的 log returns
    - initial_params: 起始參數猜測 [mu, kappa, gamma, vv, rho]
    - n_iter: MCMC 迭代次數
    - n_particles: 每次粒子濾波使用的粒子數 (越多越準但越慢)
    - proposal_std: 隨機遊走的步長 (標準差)，= 0代表固定參數
    """
    
    # 初始化
    current_params = np.array(initial_params, dtype=float)
    n_params = len(current_params)
    
    # 如果沒給步長，給一個默認的 (這通常需要根據參數大小調整)
    if proposal_std is None:
        # [mu, kappa, gamma, vv, rho] 的跳躍步伐大小
        proposal_std = np.array([0.01, 0.1, 0.002, 0.01, 0.05])
        
    else:
        proposal_std = np.array(proposal_std) # 加這行保險
        
        
    adaptable_index = np.where(proposal_std > 0)[0]
    proposal_std_reduced = proposal_std[adaptable_index]
    
    # 儲存結果的容器
    trace = np.zeros((n_iter, n_params))
    accept_count = 0
    
    # Step A: 計算起始點的 Log Likelihood
    # 這裡呼叫你的粒子濾波器
    _, current_log_lik = hm_particle_filter(returns, current_params, N=n_particles, dt=dt)
    
    # 加上 Prior (雖然初始值通常在範圍內，但習慣上加上)
    current_log_post = current_log_lik + get_log_prior(current_params)
    iterator = range(n_iter)
    if verbose:
        iterator = tqdm(iterator, desc="PMCMC Chain", leave=True) # leave=False 跑完自動消除
    # --- MCMC Loop ---
    print(f"開始 PMCMC 迭代 ({n_iter} 次)...") if verbose else None
    for i in iterator:
        
        # Step B: 提出新參數 (Proposal Step)
        # 使用隨機遊走: New = Old + Gaussian Noise
        proposal_params = current_params.copy()
        proposal_params[adaptable_index] += np.random.normal(0, proposal_std_reduced)
        
        # Step C: 計算新參數的 Log Posterior
        prior_score = get_log_prior(proposal_params)
        
        if prior_score == -np.inf:
            # 如果跑出邊界，直接拒絕 (Reject)
            proposal_log_post = -np.inf
        else:
            # 如果在邊界內，跑粒子濾波算 Likelihood
            _, proposal_log_lik = hm_particle_filter(returns, proposal_params, N=n_particles, dt=dt)
            proposal_log_post = proposal_log_lik + prior_score
            
        # Step D: Metropolis-Hastings 接受準則
        # 計算接受率 log(alpha) = log(P_new) - log(P_old)
        # 這裡假設 Proposal 是對稱的 (Gaussian)，所以不用算 q(x|y)
        log_alpha = proposal_log_post - current_log_post
        
        # 接受判斷: log(u) < log_alpha
        # np.log(np.random.rand()) 產生 (0, 1) 之間的隨機數的 log
        if np.log(np.random.rand()) < log_alpha:
            # === 接受 (Accept) ===
            current_params = proposal_params
            current_log_post = proposal_log_post
            accept_count += 1
        else:
            # === 拒絕 (Reject) ===
            # current_params 保持不變
            pass
            
        # 記錄軌跡
        trace[i, :] = current_params
        
    acceptance_rate = accept_count / n_iter
    print(f"迭代完成！接受率 (Acceptance Rate): {acceptance_rate:.2%}") if verbose else None
    return trace, acceptance_rate

def hm_params_final(Xt, dt=1/252, n_particles=888, n_iter=220, burn_in=120, prop_std=None, verbose=True):
    """
    配合不同參數使用最適合的參數估計法的Hestom參數估計最終版本
    Xt: log_return
    dt: eg.1/252年
    n_particles: particle filter還有PMCMC的粒子數量
    n_iter: PMCMC迭代次數
    mu, kappa, gamma, vv, rho
    mu: rolling
    kappa: PMCMC
    gamma: filtered
    vv: filtered
    rho:PMCMC
    """
    params_rolling = hm_params_from_rolling(Xt, h=42, delta=dt) 
    filtered_var, _ = hm_particle_filter(Xt, params_rolling, N=n_particles, dt=dt)
    params_filtered = hm_params_from_filtered(Xt,filtered_var)
    # [mu, kappa, gamma, vv, rho] 的跳躍步伐大小
    # params_filtered[0] = params_rolling[0]
    if prop_std is None:#最適解
        prop_std = np.array([0.0, 1.0, 0.0, 0.04, 0.10])
    else:
        prop_std = np.array(prop_std)
    trace, _ = run_pmcmc(
        Xt, 
        params_filtered, 
        n_iter, 
        n_particles, 
        proposal_std=prop_std,
        dt = dt,
        verbose=verbose
    )
    final_trace = trace[burn_in:]
    params_pmcmc = np.mean(final_trace,axis=0)
    return params_pmcmc



def dejdm_params_final(log_returns,dt):
    """
    DEJDM參數估計最終版本
    Xt: log_return
    dt: eg.1/252年
    """
    mu_hat_g = np.mean(log_returns) / dt
    sigma2_hat_g = np.var(log_returns) / dt
    sigma_hat_g = np.sqrt(sigma2_hat_g)
    print(f'估計的平均(𝜇)與標準差(𝜎)為 {mu_hat_g:.3%}, {sigma_hat_g:.3%}')
    
    
    def g_fun(alpha, mu, sigma, dt):
        return mu * dt + sigma * np.sqrt(dt) * norm.ppf(alpha)

    def skku(alpha, log_returns, mu, sigma, dt):
        a1, a2 = alpha

        g1 = g_fun(a1, mu, sigma, dt)
        g2 = g_fun(a2, mu, sigma, dt)

        njump_returns = log_returns[(log_returns < g1) & (log_returns > g2)]
        if len(njump_returns) < 5:
            return 10
        #     sk = skew(njump_returns, bias=False)
        #     ku = kurtosis(njump_returns, fisher=False, bias=False)
        sk = skew(njump_returns, bias=True)
        ku = kurtosis(njump_returns, fisher=False, bias=True)

        #if np.isnan(sk) or np.isnan(ku):
         #   return 100

        error = sk**2 + (ku - 3)**2
        return error
    
    
    def intensity_neg_log_likelihood(params, jump_times):
        """
        計算自激過程的負對數似然函數 (Negative Log-Likelihood)
        params: [alpha, theta, eta]
        jump_times: 跳躍發生的時間點序列
        """
        alpha, theta, eta = params
        
        # 參數必須為正，否則返回無限大 (懲罰項)
        if alpha <= 0 or theta <= 0 or eta <= 0:
            return 1e10
        
        n = len(jump_times)
        ll = 0 # Log-likelihood
        
        # 初始化：假設初始強度為長期均值 theta
        # lambda(t_0)
        lambda_current = theta 
        
        # 遍歷每一個跳躍時間間隔
        # 注意：我們從第一個跳躍開始算
        for k in range(1, n):
            t_prev = jump_times[k-1]
            t_curr = jump_times[k]
            interval = t_curr - t_prev
            
            # 1. 計算跳躍發生「前一瞬間」的強度 lambda(t_k-)
            # 強度隨時間指數衰減
            lambda_minus = theta + (lambda_current - theta) * np.exp(-alpha * interval)
            
            # 2. 計算這段時間的積分項 (Integral of lambda(s) ds)
            # 這是公式 (4.13) exp(...) 裡面的積分部分
            integral_term = theta * interval + (lambda_current - theta) * (1 - np.exp(-alpha * interval)) / alpha
            
            # 3. 累加 Log-Likelihood
            # ln f = ln(lambda(t-)) - int(lambda(s)ds)
            # 加上極小值 1e-10 防止 log(0)
            ll += np.log(lambda_minus + 1e-10) - integral_term
            
            # 4. 更新強度：跳躍發生後，強度瞬間增加 eta
            lambda_current = lambda_minus + eta
            
        return -ll # minimize 函數需要求最小值，所以取負號

    SQRT_2_PI = np.sqrt(2 * np.pi)
    
    def price_neg_log_likelihood_fast(params, returns, is_jump_mask, lambda_series, dt):
        mu, sigma, p, rho_plus, rho_minus = params

        # --- [優化 1] 快速失敗檢查 (避免無效計算) ---
        # 雖然有 bounds，但有時數值微分會稍微越界探測
        if sigma <= 1e-7 or p <= 0 or p >= 1 or rho_plus <= 1.001 or rho_minus >= -1.001:
            return 1e10
    
        # 1. Compensator (向量運算前的純量計算，極快)
        # 加上 1e-8 防止分母為 0 (雖然 bounds 擋住了，但為了保險)
        compensator_coef = p / (rho_plus - 1) + (1 - p) / (rho_minus - 1)
        
        # 2. Drift term (向量運算)
        # sigma**2 只算一次
        var_term = 0.5 * sigma**2
        drift_series = (mu - var_term - compensator_coef * lambda_series) * dt
        
        # 3. Centered Returns
        z = returns - drift_series
        
        # 準備變數
        sqrt_dt = np.sqrt(dt)
        vol_dt = sigma * sqrt_dt
        inv_vol_dt = 1.0 / vol_dt  # 乘法比除法快
        var = vol_dt**2
        
        # --- [優化 2] 處理無跳躍日子 (使用 numpy 原生函數) ---
        z_no_jump = z[~is_jump_mask]
        
        # 手動寫 norm pdf: (1/sqrt(2pi)) * exp(-0.5 * x^2)
        # 這比 call 函數快
        standardized_z_no = z_no_jump * inv_vol_dt
        pdf_no_jump = (1.0 / (vol_dt * SQRT_2_PI)) * np.exp(-0.5 * standardized_z_no**2)
        
        # 加上 1e-100 防止 log(0)
        ll_no_jump = np.sum(np.log(pdf_no_jump + 1e-100))
        
        # --- [優化 3] 處理跳躍日子 (使用 ndtr 加速) ---
        z_jump = z[is_jump_mask]
        
        if len(z_jump) > 0:
            # 預計算指數項
            # Term 1 (Positive)
            arg1 = (z_jump - rho_plus * var) * inv_vol_dt
            # 使用 ndtr 代替 norm.cdf
            term1 = p * rho_plus * np.exp(0.5 * (rho_plus * vol_dt)**2 - rho_plus * z_jump) * ndtr(arg1)
            
            # Term 2 (Negative)
            arg2 = (z_jump - rho_minus * var) * inv_vol_dt
            # 注意: 1 - ndtr(x) = ndtr(-x)
            term2 = -(1-p) * rho_minus * np.exp(0.5 * (rho_minus * vol_dt)**2 - rho_minus * z_jump) * ndtr(-arg2)
            
            pdf_jump = term1 + term2
            ll_jump = np.sum(np.log(pdf_jump + 1e-100))
        else:
            ll_jump = 0
            
        return -(ll_no_jump + ll_jump)
    
    #1, 估計jump
    alpha0 = np.array([0.85, 0.15])

    # 邊界避免極值
    # b = (1e-4, 1 - 1e-4)
    # bnds = [b,b]
    sol = minimize(
        skku,
        alpha0,
        args=(log_returns, mu_hat_g, sigma_hat_g, dt),
        method="Nelder-Mead",#L-BFGS-B,SLSQP,Nelder-Mead
    #     bounds=bnds,
    )
    g1 = g_fun(sol.x[0], mu_hat_g, sigma_hat_g, dt)
    g2 = g_fun(sol.x[1], mu_hat_g, sigma_hat_g, dt)
    up_list = np.where(log_returns > g1)[0]
    down_list = np.where(log_returns < g2)[0]
    
    # 2. 估計 alpha, theta, eta
    jump_indices = np.sort(np.concatenate((up_list, down_list)))
    # 假設 t=0 是數據的開始，dt = 1/252
    jump_times = jump_indices * dt 
    # jump_sizes = log_returns[jump_indices]

    initial_guess = [15.0, 3.0, 10.0]
    res_intensity = minimize(
        intensity_neg_log_likelihood, 
        initial_guess, 
        args=(jump_times,),
        method='Nelder-Mead',
    #     options={'maxiter': 2000}
    )
    alpha_est, theta_est, eta_est = res_intensity.x
    
    # 3. 估計price參數
    jumps_up = log_returns[up_list]
    jumps_down = log_returns[down_list] # 注意這包含負號
    p_hat = len(jumps_up) / (len(jumps_up) + len(jumps_down))
    
    rho_plus_hat = 1 / np.mean(jumps_up)
    rho_minus_hat = 1 / np.mean(np.abs(jumps_down)) 
    rho_minus_hat = - np.abs(rho_minus_hat)
    # sigma_est = sigma_hat_g
    # mu_est = mu_hat_g


    # 建立lambda序列
    n_days = len(log_returns)
    # time_points = np.arange(n_days) * dt
    
    # 把跳躍時間轉成 set 為了快速查找 (使用整數 index)
    jump_indices_set = set(jump_indices)
    # 初始化每一天的強度 lambda_series
    lambda_series = np.zeros(n_days)
    
    lambda_series[0] = theta_est + eta_est if 0 in jump_indices_set else theta_est# 初始值
    
    for t_idx in range(1, n_days):
        decay_factor = np.exp(-alpha_est * dt)
        lambda_minus = theta_est + (lambda_series[t_idx-1] - theta_est) * decay_factor
        
        # 更新今天的強度
        if t_idx in jump_indices_set:
            # 今天發生跳躍，強度飆升
            lambda_series[t_idx] = lambda_minus + eta_est
        else:
            lambda_series[t_idx] = lambda_minus

    rho_plus_init = 1.1 if np.isnan(rho_plus_hat) else rho_plus_hat
    rho_minus_init = -1.1 if np.isnan(rho_minus_hat) else rho_minus_hat
#     if rho_plus_init <= 1: rho_plus_init = 2.0
    
    initial_guess_price = [mu_hat_g, sigma_hat_g, p_hat, rho_plus_init, rho_minus_init]
#     initial_guess_price = [mu_hat_g, sigma_hat_g, p_hat, rho_plus_hat, rho_minus_hat]
    print(initial_guess_price)
    
    bounds = [
        (None, None),      # mu
        (1e-4, None),      # sigma (不要設太小 1e-6，容易造成梯度不穩定)
        (0.01, 0.99),      # p (避開 0 和 1)
        (1.1, None),       # rho+ (避開 1)
        (None, -1.1)       # rho- (避開 -1)
    ]
    
    is_jump_mask = np.zeros(n_days, dtype=bool)
    is_jump_mask[jump_indices] = True

    # 注意 options 中的 ftol 設定
    res_price = minimize(
        price_neg_log_likelihood_fast,
        initial_guess_price,
        args=(log_returns, is_jump_mask, lambda_series, dt),
        bounds=bounds,
        method='L-BFGS-B', # L-BFGS-B 通常比 SLSQP 快一點點
        options={'maxiter': 500, 'ftol': 1e-5, 'disp': True} # ftol 改成 1e-6 足夠了
    )

    mu_est, sigma_est, p_est, rho_plus_est, rho_minus_est = res_price.x

    return alpha_est, theta_est, eta_est, mu_est, sigma_est, p_est, rho_plus_est, rho_minus_est



def simulate_ogata_with_sizes(alpha,theta,eta,p,rho_plus,rho_minus, T_horizon):
    """
    改良版: 同時回傳跳躍時間與跳躍幅度
    使用 Ogata's Thinning Algorithm 模擬自激過程的跳躍時間
    params: {alpha, theta, eta}
    T_horizon: 模擬總時長 (年)
    """
    
    t_current = 0
    lambda_current = theta
    
    # 儲存結果
    jump_times = []
    jump_sizes = [] # 新增: 存幅度
    
    while t_current < T_horizon:
        # 1. 設定上限
        lambda_star = lambda_current
        
        # 2. 生成候選時間
        u = np.random.uniform(0, 1)
        # 防止 lambda_star 太小導致除以 0
        if lambda_star < 1e-6: lambda_star = 1e-6
        tau = -np.log(u) / lambda_star
        
        t_proposed = t_current + tau
        
        if t_proposed > T_horizon:
            break
            
        # 3. 計算真實強度 (衰減)
        lambda_true = theta + (lambda_current - theta) * np.exp(-alpha * tau)
        
        # 4. 接受/拒絕
        d = np.random.uniform(0, 1)
        if d <= lambda_true / lambda_star:
            # === [接受] ===
            jump_times.append(t_proposed)
            
            # --- 新增: 生成這次的跳躍幅度 J ---
            # 決定方向
            is_up = np.random.uniform(0, 1) < p
            if is_up:
                J = np.random.exponential(1/rho_plus)
            else:
                J = -np.random.exponential(1/abs(rho_minus)) # 負跳躍
            jump_sizes.append(J)
            # -------------------------------
            
            lambda_current = lambda_true + eta
            t_current = t_proposed
        else:
            # === [拒絕] ===
            lambda_current = lambda_true
            t_current = t_proposed
            
    return np.array(jump_times), np.array(jump_sizes)


def algo_ogata_thin(alpha,theta,eta,p,rho_p,rho_m, steps=252, dt=1/252):
    """
    回傳lambda的積分以及J的累加
    強制每一個區間只能夠跳一次，所以不適合模擬dt太大的情況。
    使用 Ogata's Thinning Algorithm 模擬自激過程的跳躍時間
    params: {alpha, theta, eta}
    """
    
    t_current = 0
    lambda_current = theta
    J_arr = np.zeros(steps+1)# jump
    lambda_cum = np.zeros(steps+1)# lambda積分
    
    while t_current < steps:
        lambda_star = lambda_current
        # 2. 生成候選時間
        u = np.random.uniform(0, 1)
        # 防止 lambda_star 太小導致除以 0
        if lambda_star < 1e-6: lambda_star = 1e-6
        tau = (-np.log(u) / lambda_star) / dt# 單位：日
        
        t_proposed = t_current + tau
        if t_proposed > steps:
            break
        num_steps = int(t_proposed) - int(t_current)
        
        dts = np.arange(1, num_steps+1) * dt
        lambda_cum[int(t_current)+1:int(t_proposed)+1] = lambda_cum[int(t_current)] + theta * dts + (lambda_current-theta)/alpha * (1-np.exp(-alpha*dts))
        
        
        #3.到跳躍之前的lambda積分以及lambda
        lambda_int_temp = lambda_cum[int(t_current)] + theta * (tau*dt) + (lambda_current-theta)/alpha * (1-np.exp(-alpha*(tau*dt)))
        lambda_true = theta + (lambda_current - theta) * np.exp(-alpha * (tau*dt))
        tau2 = 1 - tau % 1#距離下一整數時間點的時間

        # 4. 接受/拒絕
        d = np.random.uniform(0, 1)
        if d <= lambda_true / lambda_star:
            # === [接受] ===
            # 決定方向
            is_up = np.random.uniform(0, 1) < p
            if is_up:
                J = np.random.exponential(1/rho_p)
            else:
                J = -np.random.exponential(1/abs(rho_m)) # 負跳躍
            
            J_arr[int(t_proposed)+1] = J
            
            lambda_temp = lambda_true + eta
            #下一整數時間點的lambda
            lambda_current = theta + (lambda_temp - theta) * np.exp(-alpha * (tau2*dt))
            t_current = int(t_proposed)+1
            lambda_cum[t_current] = lambda_int_temp + theta * (tau2*dt) + (lambda_temp-theta)/alpha * (1-np.exp(-alpha*(tau2*dt)))

        else:
            # === [拒絕] ===
            lambda_temp = lambda_true
            lambda_current = theta + (lambda_temp - theta) * np.exp(-alpha * (tau2*dt))
            t_current = int(t_proposed)+1
            lambda_cum[t_current] = lambda_int_temp + theta * (tau2*dt) + (lambda_temp-theta)/alpha * (1-np.exp(-alpha*(tau2*dt)))

    if t_current != steps:
        num_steps = steps - int(t_current)
        dts = np.arange(1, num_steps+1) * dt
        lambda_cum[int(t_current)+1:] = lambda_cum[int(t_current)] + theta * dts + (lambda_current-theta)/alpha * (1-np.exp(-alpha*dts))
         
    lambda_diff = np.diff(lambda_cum)
    
    return lambda_diff, J_arr[1:]


def dejdm_simulation_single(alpha, theta, eta, mu, sigma, p, rho_p, rho_m, dt=1/252, state_n=500, steps=252, seed=None, keep_one=False):
    """
    DEJDM
    模型參數(年化)：
    - state_n: 模擬路徑數
    - steps: 每條路徑的時間步數
    - dt: 一步是幾年，ex天是1/252
    X: log return
    """
    if seed is not None:
        np.random.seed(seed)
        
    lambda_arr, J_arr = [], []
    for _ in range(state_n):
        ld, Ja = algo_ogata_thin(alpha,theta,eta,p,rho_p,rho_m, steps, dt)
        lambda_arr.append(ld)
        J_arr.append(Ja)

    lambda_int = np.array(lambda_arr)
    J_sum = np.array(J_arr)

    dWt = np.random.normal(size=(state_n,steps)) * np.sqrt(dt)
                                                                   
    Ek = p/(rho_p-1) + (1-p)/(rho_m-1)
    
    growth = np.exp(np.cumsum((mu-0.5*sigma**2)*dt - Ek*lambda_int + sigma*dWt + J_sum,axis=1))
    if keep_one:
        growth = np.hstack([np.ones((state_n,1)),growth])
    return growth



#%% dejdm
alpha, theta, eta, mu, sigma, p, rho_p, rho_m = (14.1,4.0,11.8,0.11,0.11,0.40,33.3,-37.4)
Sp = dejdm_simulation_single(alpha, theta, eta, mu, sigma, p, rho_p, rho_m, dt=1/252, state_n=7, steps=252*3, seed=None, keep_one=True)
plt.plot(Sp.T)


#%% fbm
if __name__ == "__main__":
    # 測試代碼範例
    mu = [0.05, 0.05]
    cov = [[0.01, 0.016], 
                [0.016, 0.04]]
    
    
    # 設定兩支股票相關係數為 0.8
    corr = [[1.0, 0.4], 
                [0.4, 1.0]]
    
    H_list = [0.6,0.4]
    growth = fbm_simulation(mu,cov,corr,H_list,state_n=5000,steps=252,cor_open=False)
    
    growth = fbm_simulation_end(mu,cov,corr,H_list,state_n=5000,T=252,cor_open=False)

                            

#%% heston model 
if __name__ == "__main__":
    
    # 測試代碼範例
    mu = [0.05, 0.05]
    kappa = [2.0, 2.0]
    gamma = [0.04, 0.04]
    vv = [0.3, 0.3]
    rho = [-0.2, -0.3]
    # 設定兩支股票相關係數為 0.8
    corr_mat = [[1.0, 0.8], 
                [0.8, 1.0]]
    
    g, v, x = hm_simulation(mu, kappa, gamma, vv, rho, corr_mat, state_n=10000, steps=252)
    x = x[...,1:]
    v = np.diff(v,axis=2)
    
    # 驗證
    # 取出第一步的 return 來看相關性
    corr_measured = np.corrcoef(x[:, 0, 0], x[:, 1, 0])
    print("設定相關係數: 0.8")
    print(f"模擬實測相關係數: {corr_measured[0,1]:.4f}")
        
    print("\n--- A. 驗證股票與自身的波動率相關係數 (Leverage Effect) ---")
    # 我們取第 0 個時間步 (第一天的模擬結果)
    # 因為初始值是常數，第一天的數值完全由隨機項 Wt 決定，最能反映 Rho
    for i in range(2):
        # 取出第 i 支股票的所有路徑數據
        stock_ret = x[:, i, -1]  # Log Return
        stock_vol = v[:, i, -1]  # Variance (Vt)
        
        # 計算相關係數
        corr = np.corrcoef(stock_ret, stock_vol)[0, 1]
        
        print(f"Stock {i}: 設定 Rho = {rho[i]:.2f} | 實測 Corr = {corr:.4f}")
        
        # 簡單判斷
        if abs(corr - rho[i]) < 0.05:
            print(f"  -> ✅ Stock {i} 驗證通過")
        else:
            print(f"  -> ❌ Stock {i} 偏差過大")
    
    print("\n--- B. 驗證波動率之間的獨立性 (Vol-Vol Correlation) ---")
    # 根據模型假設，不同股票的波動率隨機項應該是獨立的 (除非你有額外設定)
    # 你的 big_corr 右下角是 Identity Matrix，所以理論值應為 0
    vol_corr = np.corrcoef(v[:, 0, 0], v[:, 1, 0])[0, 1]
    print(f"Vol 1 vs Vol 2: 理論值 = 0.00 | 實測 Corr = {vol_corr:.4f}")




#%% PMCMC
if __name__ == '__main__':
    # --- 設定環境與數據 ---
    # 假設你已經有了 returns_obs (真實 Heston 生成的數據)
    # true_params = [0.05, 1.8, 0.0225, 0.10, -0.8]
    
    # 真實參數
    mu = 0.05
    kappa = 1.8
    gamma = 0.0225
    vv = 0.10
    rho = -0.8
    N = 1
    T = 10
    steps = 252*T
    true_params = [mu, kappa, gamma, vv, rho] 
    
    # 產生數據
    growth, Vt_true, Xt_path = hm_single(mu, kappa, gamma, vv, rho, T, N, steps, seed=2, keep_one=True)
    returns_obs = Xt_path.flatten() # 這是 x_1 ... x_T
    real_vol_path = Vt_true.flatten()
    
    # 1. 給一個初始猜測 (可以使用 Rolling Window 的結果，或隨便猜一個合理的)
    # 這裡故意給一個稍微有點歪的初始值，看它能不能修回來
    initial_guess = [0.02, 1.0, 0.04, 0.2, -0.2] 
    
    # 2. 設定步長 (Tuning)
    # 這是 MCMC 最需要調的地方。步長太大，接受率會很低；步長太小，收斂很慢。
    # 經驗法則：目標接受率大約在 20% - 40% 之間
    prop_std = np.array([0.02, 0.2, 0.005, 0.02, 0.1]) 
    prop_std = np.array([0.004, 0.05, 0.001, 0.004, 0.02])
    # 3. 執行 PMCMC
    # 建議 N_particles=500~1000 (太少 Likelihood 雜訊大，太多跑太慢)
    # 建議 n_iter=2000 (前 500-1000 次通常是 Burn-in)
    trace, acc_rate = run_pmcmc(
        returns_obs, 
        initial_guess, 
        n_iter=3000, 
        n_particles=600, 
        proposal_std=prop_std
    )
    
    # --- 4. 結果視覺化 (Trace Plot) ---
    param_names = ['Mu', 'Kappa', 'Gamma', 'V_v', 'Rho']
    true_vals = [0.05, 1.8, 0.0225, 0.10, -0.8]
    
    plt.figure(figsize=(12, 10))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(trace[:, i], label='MCMC Trace')
        plt.axhline(y=true_vals[i], color='r', linestyle='--', label='True Value')
        plt.title(f'Trace of {param_names[i]}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- 5. 去除 Burn-in 後的估計值 ---
    burn_in = 1000 # 假設前 1000 次還在找路
    final_trace = trace[burn_in:]
    
    print("\n=== PMCMC 估計結果 (Posterior Mean) ===")
    means = np.mean(final_trace, axis=0)
    stds = np.std(final_trace, axis=0)
    
    df_res = pd.DataFrame({
        'Parameter': param_names,
        'True Value': true_vals,
        'PMCMC Mean': means,
        'Posterior Std': stds,
        'Error': means - true_vals
    })
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_res)




#%% hm --- 測試與驗證 ---
if __name__ == "__main__":
    # 1. 準備數據 (使用你上一段程式碼產生的真實數據)
    # 假設 hm_single 已經跑過，我們有 x (log returns) 和 Vt_true
    # 這裡重新跑一次確保變數存在
    
    # 真實參數
    mu = 0.05
    kappa = 1.8
    gamma = 0.0225
    vv = 0.10
    rho = -0.8
    N = 1
    T = 10
    steps = 252*T
    true_params = [mu, kappa, gamma, vv, rho] 
    
    # 產生數據
    growth, Vt_true, Xt_path = hm_single(mu, kappa, gamma, vv, rho, T, N, steps, seed=2, keep_one=True)
    returns_obs = Xt_path.flatten() # 這是 x_1 ... x_T
    real_vol_path = Vt_true.flatten()
    
    # 2. 執行粒子濾波器
    # 假設我們知道參數 (先用真實參數測試濾波效果)
    print("開始執行粒子濾波 (N=1000)...")
    filtered_var, log_lik = hm_particle_filter(returns_obs, true_params, N=1000, dt=1/252)
    
    # 3. 繪圖比較 (重現教科書 Fig 3.2)
    # plt.figure(figsize=(12, 6))
    # plt.plot(real_vol_path, color='green', alpha=0.5, label='Real Variance (Hidden)')
    # plt.plot(filtered_var, color='red', linewidth=1, label='Filtered Variance (Particle Filter)')
    # plt.title(f'Particle Filter Result (Log Likelihood: {log_lik:.2f})')
    # plt.legend()
    # plt.show()
    
    # 4. 驗證準確度
    corr_filter = np.corrcoef(real_vol_path, filtered_var)[0,1]
    mse_filter = np.mean((real_vol_path - filtered_var)**2)
    print(f"濾波結果與真實變異數的相關係數: {corr_filter:.4f}")
    print(f"均方誤差 (MSE): {mse_filter:.6f}")
    
    
    print(real_vol_path[:5],filtered_var[:5])
    print('filter corr',np.corrcoef(np.diff(filtered_var), returns_obs[1:])[0,1])
    print('true corr',np.corrcoef(np.diff(real_vol_path), returns_obs[1:])[0,1])
    print(np.mean(returns_obs*252))
    
    
    # --- 使用範例 ---
    params1 = hm_params_from_rolling(returns_obs, h=42, delta=1/252)
    params2 = hm_params_from_filtered(returns_obs, filtered_var, delta=1/252)
    print('rolling',np.round(params1,3))
    print('filtered',np.round(params2,3))
    print('true',true_params)
 

#%% hm 多筆
if __name__ == '__main__':    
    import pandas as pd
    tickers = ['2330.TW','2303.TW']
    df1 = pd.read_parquet(rf'C:\上課講義\自研\財工project\data\dfdic\台股\上市\{tickers[0]}.parquet').close
    df2 = pd.read_parquet(rf'C:\上課講義\自研\財工project\data\dfdic\台股\上市\{tickers[1]}.parquet').close
    price = pd.concat([df1,df2],axis=1)
    price.columns = tickers
    Xt = np.log(price).diff(axis=0).dropna()
    filtered_vars = []
    for i in range(len(tickers)):
        rolling_params = hm_params_from_rolling(Xt.iloc[:,i], h=42, delta=1/252)
        filtered_var, _ = hm_particle_filter(Xt.iloc[:,i], rolling_params, N=1000, dt=1/252)
        filtered_vars.append(filtered_var)
    
    
#%%
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    os.chdir(r'D:\上課講義\自研\財工project\dfdic')
    tickers = ['2330.TW','2303.TW']
    
    df = pd.read_parquet(f'台股/{tickers[0]}.parquet')
    price = df.loc[:,['close']]
    price.columns = [tickers[0]]
    

    # for symbol in tickers[1:]:
    #     df = pd.read_parquet(f'台股/{symbol}.parquet')
    #     price.loc[:,symbol] = df.loc[:,'close']

    price.dropna(inplace=True)
    price1 = price.iloc[:1500]#train
    price2 = price.iloc[1500:1800]#test
    ST = price1.to_numpy()
    
    wn = 50
    logr = np.log(price1).diff().dropna()
    Vt = logr.rolling(wn).var().dropna()
    Vt2 = logr.rolling(wn,center=True).var().dropna()

    logr_bar = logr.rolling(wn).mean().dropna()
    
    Vt['vt2'] = Vt2
    Vt.plot()
    Vt2.plot()
    
    
#%%
if __name__ == "__main__":
    
    N = 1
    steps = 252
    T = 1
    
    S, v = hm_single(S0,v0,mu,vol_of_vol,kappa,rho,theta,T,N,steps)
    S = S.flatten()
    v = v.flatten()
    plt.plot(v)
    
    Xt = np.log(S)
    Vt = logr.rolling(wn).var().dropna()
    plt.plot(Vt)

    
    #%%
    # fbm_simulation(cov_matrix,dt,D,H_list,state_n=5000,steps=252,T=1)
    S0_list = [10]*10
    H_list = [0.5]*10
    
    
    #%% BM
    col_mean = np.array([0.08, 0.12])
    cov_matrix = np.array([
    [0.04,   0.03],
    [0.03, 0.0625]
    ])
    corr_matrix = np.array([
    [1.0, 0.6],
    [0.6, 1.0]
    ])
    
    bm_end = bm_simulation_end(col_mean,cov_matrix,corr_matrix,state_n=3000000,T=10,vrt='',cor_open=True,seed=1)
    bm_s = bm_simulation(col_mean,cov_matrix,corr_matrix,state_n=3000000,steps=10,vrt='',seed=1,keep_one=False)
    
    print(np.mean(bm_end,axis=0), np.mean(bm_s[...,-1],axis=0))
    print(np.std(bm_end,axis=0), np.std(bm_s[...,-1],axis=0))
    
    
    #%%
    br = bm_simulation_single(mu=0.05,sigma=0.1,state_n=5000,steps=252,T=1,vrt='')
    plt.plot(br[1,:])    
    
    #%% Hestpn
    mu = 0.05
    kappa = 1.8
    gamma = 0.0225
    vv = 0.10
    rho = -0.5
    N = 1
    T = 10
    steps = 252*T

    growth, Vt, x=  hm_single(mu, kappa, gamma, vv, rho, T, N, steps)
    x = x.flatten()
    
    Vt = Vt.flatten()
    
    dVt = np.diff(Vt)
    cor = np.corrcoef(x[1:],dVt)
    print(cor)


    # 估計參數
    vt = pd.Series(x).rolling(35,center=True).var().dropna()*252
    dvt = vt.diff().dropna()
    
    
    cor = np.corrcoef(x[dvt.index],dvt)
    print(cor)
    
    
    
    # cor = np.corrcoef(vt,Vt[vt.index])
    # print(cor)    
    
    # cor = np.corrcoef(dvt,dVt[dvt.index])
    # print(cor)    
    
    



    # vt.plot()      
    # plt.figure()
    # pd.Series(Vt.flatten()).plot()
    # plt.plot()
    # plt.plot(Vt.flatten())
    
    
    
    
    