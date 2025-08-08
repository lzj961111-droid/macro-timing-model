#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:56:42 2025

@author: zhengjunli
"""

import pandas as pd
import numpy as np
from WindPy import w
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

def construct_macro_features(df, macro_factors, trade_dates, halflife_dict, decay_days=23):
    """
    构建加入衰减后的宏观因子特征。
    
    参数：
        df: 原始日度数据，index 为交易日。
        macro_factors: 宏观数据，index 为发布日期，包含列 ['类型', '今值']。
        trade_dates: 所有交易日列表。
        halflife_dict: 每个因子的半衰期字典，例如 {'PMI': 4, 'PPI': 6}。
        decay_days: 衰减延续的最大天数（如23交易日）。
        
    返回：
        带有衰减因子的 DataFrame。
    """
    df = df.copy()
    
    for factor in halflife_dict.keys():
        half_life = halflife_dict[factor]
        lambda_ = np.log(2) / half_life
        decay_col = f'{factor}_decay'
        df[decay_col] = np.nan
        
        factor_data = macro_factors[macro_factors['类型'] == factor].copy()
        # 将发布日期映射为最近一个交易日
        factor_data.index = [trade_dates[trade_dates <= d][-1] for d in factor_data.index]
        
        for date, value in factor_data['今值'].dropna().items():
            if date not in df.index:        # ★ 新增：事件日期不在当前窗口，跳过
                continue
            start_idx = df.index.get_loc(date)
            for i in range(decay_days):
                if start_idx + i >= len(df):
                    break
                t = df.index[start_idx + i]
                decay_value = value * np.exp(-lambda_ * i)
                
                if pd.isna(df.at[t, decay_col]):
                    df.at[t, decay_col] = decay_value
                else:
                    df.at[t, decay_col] = max(df.at[t, decay_col], decay_value)
    
    return df

plt.rcParams['font.family'] = 'Kaiti SC'  # 微软雅黑，适用于Windows和Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

w.start()

rolling_days = 10

macro_factors = pd.read_excel('macro_15_to_25 2.xlsx',index_col=0)
macro_factors.index = pd.to_datetime(macro_factors.index)

cpi = macro_factors[macro_factors['类型']=='CPI']
start = '2014-12-31'
end = '2025-08-04'

factors = ['PMI','PPI','CPI','新增人民币贷款', 'M2：同比','社会融资规模：同比','工业增加值','固定资产投资',
           'GDP：同比', 'M0：同比', 'M1：同比','出口:当月同比']



df_close = w.wsd('881001.WI',"close",start,end,usedf=True)[1]
df_close.index = pd.to_datetime(df_close.index)
df_ret = df_close.pct_change()
df_ret = df_ret.fillna(0)

trade_dates = df_ret.index
cpi.index = [trade_dates[trade_dates <= d][-1] for d in cpi.index]

df = pd.DataFrame(index = df_close.index)
df['ret'] = df_ret['CLOSE']
# df['CPI'] = cpi['今值']

# df_test = df.copy()#尝试一下用function生成衰减因子，所以复制一个df出来

# # 设置一个临时的半衰期参数
# half_life = 4 # 单位是交易日
# lambda_ = np.log(2) / half_life
# decay_days = 23  # 衰减最大延续时间

# df['cpi_decay'] = np.nan

# # 遍历 CPI 有效值的每个发布日期（即那些不是 NaN 的日期）
# for date, value in df['CPI'].dropna().items():
    
#     # 找到该发布日期在整个交易日 index 中的位置（整数位置索引）
#     start_idx = df.index.get_loc(date)
    
#     # 向后衰减填充值，最多延续 decay_days 天
#     for i in range(decay_days):
        
#         # 防止越界：如果已经超过最后一个交易日，则跳出循环
#         if start_idx + i >= len(df):
#             break
        
#         # 当前衰减目标的日期
#         t = df.index[start_idx + i]
        
#         # 根据指数衰减公式计算当前衰减值
#         decay_value = value * np.exp(-lambda_ * i)
        
#         # 如果当前日期的 cpi_decay 为空，直接赋值
#         if pd.isna(df.at[t, 'cpi_decay']):
#             df.at[t, 'cpi_decay'] = decay_value
#         else:
#             # 如果该日期已有其他衰减值（比如多个CPI值重叠），可以选择叠加或取最大值
#             df.at[t, 'cpi_decay'] = max(df.at[t, 'cpi_decay'], decay_value) # 你也可以改成 max(df.at[t], decay_value)
            
# df[['CPI', 'cpi_decay']].plot(title='CPI 衰减填充效果')

# for factor in factors:
#     decay_col = f'{factor}_decay'
#     df[decay_col] = np.nan
#     factor_data = macro_factors[macro_factors['类型'] == factor]
#     factor_data.index = [trade_dates[trade_dates <= d][-1] for d in factor_data.index]

#     for date, value in factor_data['今值'].dropna().items():
        
#         start_idx = df.index.get_loc(date)

#         for i in range(decay_days):
#             if start_idx + i >= len(df):
#                 break

#             t = df.index[start_idx + i]
#             decay_value = value * np.exp(-lambda_ * i)

#             if pd.isna(df.at[t, decay_col]):
#                 df.at[t, decay_col] = decay_value
#             else:
#                 df.at[t, decay_col] = max(df.at[t, decay_col], decay_value)

## ✅✅✅✅尝试用function生成因子，看看是不是一样的
# 手动方式构造的列名
original_cols = [f'{factor}_decay' for factor in factors]
df_base = df.copy()

# 用函数方式重新构造
halflife_dict = {factor: 4 for factor in factors}
df = construct_macro_features(df.copy(), macro_factors, trade_dates, halflife_dict, decay_days=23)


# 初始化列表用于存储每个滞后天数下的IC
ic_list = []
max_lag = 40  # 设置最大的滞后天数（也就是未来收益窗口大小）

# # 遍历每一个未来滞后窗口（从1天到20天）
# for d in range(1, max_lag + 1):
#     x = df['cpi_decay']  # 当前因子值（在发布日期上衰减后得到）

#     # ✅ 构造目标值 y：真实的未来 d 日 simple return 累计收益（通过滚动连乘计算）
#     # shift(-1)：从 t+1 开始（因子不能预测当天）
#     # rolling(d)：往后滚 d 天
#     # apply(lambda x: np.prod(1 + x) - 1)：用 simple return 的连乘法计算总收益
#     y = df['ret'].shift(-1).rolling(d).apply(lambda x: np.prod(1 + x) - 1, raw=True)

#     # 过滤掉因子值和未来收益都为 NaN 的情况
#     valid = x.notna() & y.notna()

#     # 如果有效样本数量大于0，则计算 Spearman IC
#     if valid.sum() > 0:
#         ic = spearmanr(x[valid], y[valid])[0]
#     else:
#         ic = np.nan

#     ic_list.append(ic)  # 将结果加入列表中

# # 构造横轴为滞后天数的 array，纵轴为IC值
# lags = np.arange(1, max_lag + 1)
# ics = np.array(ic_list)

# # 设置 matplotlib 中文字体
# plt.rcParams['font.family'] = 'Kaiti SC'       # 你机器上已有的中文字体
# plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# # 画图：滞后 IC 衰减曲线
# plt.figure(figsize=(8, 4))
# plt.plot(lags, ics, marker='o')
# plt.title('滞后IC衰减曲线（真实simple return）')  # 中文标题：IC随未来窗口的变化
# plt.xlabel('未来累计收益窗口（交易日）')             # 横轴：滞后天数 d
# plt.ylabel('Spearman IC')                      # 纵轴：IC 值
# plt.grid(True)
# plt.show()



###✅✅✅✅批量画图


# max_lag = 22

# half_life_range = range(2,21)

# # factors = ['PPI']

# lag_ic_df = pd.DataFrame(index = range(1,max_lag+1),columns = factors)

# for factor in factors:
#     decay_col = f'{factor}_decay'
#     x = df[decay_col]
#     ic_list = []
#     for d in range(1,max_lag+1):
#         y = df['ret'].shift(-1).rolling(d).apply(lambda x: np.prod(1 + x) - 1, raw=True)
#         valid = x.notna() & y.notna()
#         if valid.sum()>0:
#             ic = spearmanr(x[valid],y[valid])[0]
#         else:
#             ic = np.nan
#         lag_ic_df.loc[d,factor]=ic

# df = df.fillna(0)




# 拟合效果评估
from sklearn.metrics import r2_score, mean_squared_error

# # ✅✅✅✅Elastic Net回归，初步来看结果非常不好哈哈，也是吐了

# df['ret_1_5'] = (
#     df['ret']
#     .shift(-1)  # 从 T+1 开始
#     .rolling(5)  # 向后看5天
#     .apply(lambda x: np.prod(1 + x) - 1, raw=True)  # 累计 simple return
# )

# df = df.dropna()

# # =====================
# # 【新增】标准化目标变量 ret_1_5
# # =====================
# scaler_y = StandardScaler()
# valid_y = df['ret_1_5'].notna()
# df['ret_1_5_raw'] = df['ret_1_5']  # 保留原始版本
# df.loc[valid_y, 'ret_1_5'] = scaler_y.fit_transform(df.loc[valid_y, ['ret_1_5']])

# # =====================
# # 提取特征与目标变量
# # =====================
# X = df.drop(columns=['ret', 'ret_1_5', 'ret_1_5_raw'])  # ✅【保留 ret_1_5_raw】
# y = df['ret_1_5']

# # 删除因子全为 NaN 的列
# X = X.dropna(axis=1, how='all')

# # 保留训练数据：目标值为非空的数据
# valid = y.notna()
# X_valid = X[valid]
# y_valid = y[valid]

# # =====================
# # 标准化特征变量 X（原来已有）
# # =====================
# scaler_X = StandardScaler()
# X_scaled = scaler_X.fit_transform(X_valid)

# # =====================
# # ElasticNetCV 模型
# # =====================
# model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .9, .95, .99, 1.0], alphas=[0.01, 0.1, 1, 10])
# model.fit(X_scaled, y_valid)

# # =====================
# # 模型评估（仍然在训练集上）
# # =====================
# y_pred = model.predict(X_scaled)

# # ✅ R² 依然可以比较
# r2 = r2_score(y_valid, y_pred)

# # ✅ RMSE 因为 y 已标准化，结果也在标准差单位上
# rmse = mean_squared_error(y_valid, y_pred)

# # =====================
# # 打印结果
# # =====================
# print("Best alpha:", model.alpha_)
# print("Best l1_ratio:", model.l1_ratio_)
# print("In-sample R^2:", r2)
# print("In-sample RMSE:", rmse)

# # =====================
# # 系数输出
# # =====================
# coef_df = pd.DataFrame({
#     'factor': X.columns,
#     'coefficient': model.coef_
# })
# print(coef_df)

## ✅✅✅✅单一Elastic Net模型结束，现在试试看ElasticNet+Sigmoid
## ✅ 构造分类目标变量：预测未来5日涨跌方向（1表示上涨，0表示下跌）

# df['target_return'] = df['ret'].shift(-1).rolling(5).apply(lambda x: np.prod(1 + x) - 1, raw=True)
# df['target_direction'] = (df['target_return'] > 0).astype(int)

# # ✅ 特征矩阵与目标
# X = df.drop(columns=['ret', 'target_return', 'target_direction']).copy()
# y = df['target_direction']

# # 去除缺失值
# valid = X.notna().all(axis=1) & y.notna()
# X = X[valid]
# y = y[valid]

# # 标准化因子
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ✅ 逻辑回归 + ElasticNet（使用liblinear支持l1_ratio）
# param_grid = {
#     'C': [0.01, 0.1, 1, 10],          # 相当于 alpha 的倒数
#     'l1_ratio': [0.0, 0.1, 0.5, 0.9]  # 0表示L2，1表示L1
# }

# model = LogisticRegression(
#     solver='saga',  # saga 支持 l1_ratio
#     penalty='elasticnet',
#     max_iter=10000
# )

# grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
# grid.fit(X_scaled, y)

# # 输出最佳参数与指标
# best_model = grid.best_estimator_
# print(f"Best C: {grid.best_params_['C']}")
# print(f"Best l1_ratio: {grid.best_params_['l1_ratio']}")

# y_prob = best_model.predict_proba(X_scaled)[:, 1]
# y_pred = best_model.predict(X_scaled)

# print("Accuracy:", accuracy_score(y, y_pred))
# print("ROC AUC:", roc_auc_score(y, y_prob))

# # 系数表格
# coef_df = pd.DataFrame({
#     'factor': X.columns,
#     'coefficient': best_model.coef_[0]
# })
# print(coef_df)

#✅✅✅✅再试一个模型，XGBoost模型
from xgboost import XGBClassifier

# 构造目标变量（未来5日累计收益 > 0 为1）
df['target'] = (df['ret'].shift(-1).rolling(rolling_days).apply(lambda x: np.prod(1 + x) - 1, raw=True) > 0).astype(int)
df_base['target'] = (df_base['ret'].shift(-1).rolling(rolling_days).apply(lambda x: np.prod(1 + x) - 1, raw=True) > 0).astype(int)
# # 初步实验XGBoost模型，目前来看效果比线性模型好很多很多，但有可能受到数据泄露影响
# # 特征和标签
# features = [col for col in df.columns if col.endswith('_decay')]
# X = df[features]
# y = df['target']

# # 去除NaN
# valid = X.notna().all(axis=1) & y.notna()
# X = X.loc[valid]
# y = y.loc[valid]

# # 使用时间序列交叉验证（防止数据泄露）
# tscv = TimeSeriesSplit(n_splits=5)

# # 建立XGBoost模型并网格搜索超参
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [2, 3, 4],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
# }

# xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
# grid = GridSearchCV(xgb, param_grid, scoring='roc_auc', cv=tscv, verbose=0)
# grid.fit(X, y)

# best_model = grid.best_estimator_

# # 预测与评估
# y_pred_prob = best_model.predict_proba(X)[:, 1]
# y_pred = (y_pred_prob > 0.5).astype(int)

# acc = accuracy_score(y, y_pred)
# auc = roc_auc_score(y, y_pred_prob)
# cm = confusion_matrix(y, y_pred)

# # 输出结果
# print(f"Best params: {grid.best_params_}")
# print(f"Accuracy: {acc}")
# print(f"ROC AUC: {auc}")
# print("Confusion Matrix:")
# print(cm)

# # 特征重要性
# importance = pd.DataFrame({
#     'feature': features,
#     'importance': best_model.feature_importances_
# }).sort_values(by='importance', ascending=False)

# print("\nTop features:")
# print(importance)

##✅✅✅✅XGboost模型测试结束，效果不错，下面试一下样本外的预测结果如何，有点紧张了哈哈

# # 1. 划分数据集
# total_len = len(df)
# train_end = int(total_len * 0.7)
# val_end = int(total_len * 0.85)

# train_df = df.iloc[:train_end]
# val_df = df.iloc[train_end:val_end]
# test_df = df.iloc[val_end:]

# # 2. 拆分特征与标签
# feature_cols = [col for col in df.columns if col not in ['ret', 'target']]
# X_train, y_train = train_df[feature_cols], train_df['target']
# X_val, y_val = val_df[feature_cols], val_df['target']
# X_test, y_test = test_df[feature_cols], test_df['target']

# # 合并 train 和 val 以做 GridSearchCV（cross-validation）
# X_gs = pd.concat([X_train, X_val])
# y_gs = pd.concat([y_train, y_val])

# # 3. 模型 + 网格搜索
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [2, 3, 4],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8]
# }
# xgb_clf = XGBClassifier(eval_metric='logloss')
# grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='roc_auc', verbose=0)
# grid_search.fit(X_gs, y_gs)

# print("✅ Best params:", grid_search.best_params_)

# # 4. 最终模型 on Test
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# y_proba = best_model.predict_proba(X_test)[:, 1]

# accuracy = accuracy_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_proba)
# cm = confusion_matrix(y_test, y_pred)

# print("✅ Test Accuracy:", accuracy)
# print("✅ Test ROC AUC:", auc)
# print("✅ Confusion Matrix:\n", cm)

# # 5. 特征重要性输出
# importances = best_model.feature_importances_
# feature_importance_df = pd.DataFrame({
#     'feature': feature_cols,
#     'importance': importances
# }).sort_values(by='importance', ascending=False)

# print("\nTop features:")
# print(feature_importance_df.head(10))

# # 6. 可视化混淆矩阵
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
# plt.title("Confusion Matrix (Test)")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.show()

## ✅✅✅✅遗传算法寻找最佳半衰期

import random
from sklearn.model_selection import train_test_split

# 遗传算法参数
POP_SIZE = 10
N_GEN = 10
MUTATION_RATE = 0.1
HALF_LIFE_RANGE = (2, 15)

# 示例因子名称（你可以替换为你的实际因子列表）
factor_list = factors

# Step 1: 初始化种群
def initialize_population(pop_size, factor_num):
    return [np.random.randint(HALF_LIFE_RANGE[0], HALF_LIFE_RANGE[1]+1, size=factor_num).tolist() for _ in range(pop_size)]

# Step 2: 评估适应度
def evaluate_fitness(individual, df_base, macro_factors, trade_dates, factor_list):
    half_life_dict = dict(zip(factor_list, individual))
    
    # 构造带半衰期因子的数据
    df = construct_macro_features(df_base.copy(), macro_factors, trade_dates, half_life_dict)
    
    feature_cols = [col for col in df.columns if col not in ['ret', 'target']]
    train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_val, y_val = val_df[feature_cols], val_df['target']

    model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    return auc

# Step 3: 选择 + 交叉 + 变异
def select(population, scores):
    selected = random.choices(population, weights=scores, k=len(population))
    return selected

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(individual)-1)
        individual[idx] = random.randint(HALF_LIFE_RANGE[0], HALF_LIFE_RANGE[1])
    return individual

# Step 4: 主函数
def run_genetic_optimization(df_base, macro_factors, trade_dates, factor_list):
    population = initialize_population(POP_SIZE, len(factor_list))
    best_auc = -np.inf
    best_individual = None

    for gen in range(N_GEN):
        scores = [
            evaluate_fitness(ind, df_base, macro_factors, trade_dates, factor_list)
            for ind in population
        ]
        gen_best_score = max(scores)
        gen_best_ind = population[scores.index(gen_best_score)]

        # 更新全局最优
        if gen_best_score > best_auc:
            best_auc = gen_best_score
            best_individual = gen_best_ind

        print(f"Generation {gen+1}: Best AUC = {gen_best_score:.4f}, Best Half-life = {gen_best_ind}")

        # 精英保留：将当前最优个体复制出来
        elite = gen_best_ind.copy()

        # 选择、交叉、变异，生成 len(population) - 1 个子代
        selected = select(population, scores)
        children = [
            mutate(crossover(random.choice(selected), random.choice(selected)))
            for _ in range(len(population) - 1)
        ]

        # 把最优个体加入下一代
        children.append(elite)
        population = children

    return best_individual, best_auc

# ==============================================================
#  折内 GA：给定 df_train → 返回该折最优 half-life 列表
# ==============================================================
def search_best_halflife_fold(factors, df_train, macro_factors, trade_dates):
    """
    在单个折内再次跑遗传算法，只用折内训练样本做内部 70/30。
    返回: (best_halflife, best_auc)
    """
    best_halflife, best_auc = run_genetic_optimization(
    df_base=df_train,
    macro_factors=macro_factors,
    trade_dates=trade_dates,   # ← 用该折的交易日列表
    factor_list=factors                     # ← 关键字改回 factor_list
    )   
    print(f"  ↳ fold-inner best AUC = {best_auc:.4f}")
    return best_halflife

best_halflife, best_auc = run_genetic_optimization(
    df_base=df_base,
    macro_factors=macro_factors,
    trade_dates=pd.DatetimeIndex(df_base.index),
    factor_list=factors
)

# #测试：全局半衰期优化
# if __name__ == "__main__":               # 防止被 import 时自动执行
#     # ① 重新按 best_halflife 生成日频特征
#     halflife_dict_fixed = dict(zip(factors, best_halflife))
#     df_feat = construct_macro_features(
#         df_base.copy(), macro_factors, trade_dates, halflife_dict_fixed
#     )

#     feature_cols = [f"{f}_decay" for f in factors]
#     X_all = df_feat[feature_cols].values
#     y_all = df_feat["target"].values
#     dates_all = df_feat.index

#     # ② 定义 6 折 TimeSeriesSplit（每折测试窗 ≈ 126 交易日 ≈ 半年）
#     tscv = TimeSeriesSplit(n_splits=6, test_size=126)
#     auc_scores = []

#     # ③ 逐折训练 → 评估
#     for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
#         X_tr, X_te = X_all[train_idx], X_all[test_idx]
#         y_tr, y_te = y_all[train_idx], y_all[test_idx]

#         model = XGBClassifier(
#             max_depth=5, n_estimators=200, learning_rate=0.05,
#             subsample=0.8, colsample_bytree=0.8, random_state=42
#         )
#         model.fit(X_tr, y_tr)

#         proba = model.predict_proba(X_te)[:, 1]
#         auc   = roc_auc_score(y_te, proba)
#         auc_scores.append(auc)

#         print(f"Fold {fold+1}: AUC = {auc:.4f}")

#     print(f"\nWalk-forward mean AUC = {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    

##✅✅✅✅rollingwindowsplit定义
from sklearn.model_selection import BaseCrossValidator
class RollingWindowSplit(BaseCrossValidator):
    """
    固定长度滚动窗交叉验证器
    """
    def __init__(self, train_size, test_size, step=126, start=0):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.start = start

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        train_start = self.start
        while True:
            train_end = train_start + self.train_size
            test_end  = train_end + self.test_size
            if test_end > n_samples:
                break
            train_idx = np.arange(train_start, train_end)
            test_idx  = np.arange(train_end,  test_end)
            yield train_idx, test_idx
            train_start += self.step   # 每次向前滑 step 个样本

    def get_n_splits(self, X=None, y=None, groups=None):
        n_samples = len(X)
        return max(0, (n_samples - self.train_size - self.test_size - self.start) // self.step + 1)

# ==============================================================
#  Walk-forward（6 折）— 每折内部重新寻 half-life
# ==============================================================
if __name__ == "__main__":

    feature_cols_template = [f"{f}_decay" for f in factors]
    dates_all = df_base.index
    auc_scores = []

    tscv = TimeSeriesSplit(n_splits=6, test_size=126)
    rws = RollingWindowSplit(train_size=756, test_size=126, step=126, start=0)

    for fold, (train_idx, test_idx) in enumerate(rws.split(dates_all), 1):
        train_start, train_end = dates_all[train_idx[0]], dates_all[train_idx[-1]]
        test_start,  test_end  = dates_all[test_idx[0]],  dates_all[test_idx[-1]]
        print(f"[Fold {fold}] Train {train_start.date()} → {train_end.date()} | "
              f"Test {test_start.date()} → {test_end.date()}")

    # for fold, (train_idx, test_idx) in enumerate(tscv.split(dates_all)):
    #     print(f"\n=== Fold {fold+1} ===")

        # ① 取该折的训练 / 测试 dataframe
        df_train = df_base.iloc[train_idx].copy()
        df_test  = df_base.iloc[test_idx].copy()

        # ② 折内 GA  → half-life
        best_halflife = search_best_halflife_fold(
            factors, df_train, macro_factors, trade_dates
        )
        hl_dict = dict(zip(factors, best_halflife))

        # ③ 用折内 half-life 重新构造特征
        df_train_feat = construct_macro_features(
            df_train.copy(), macro_factors, trade_dates, hl_dict
        )
        df_test_feat = construct_macro_features(
            df_test.copy(), macro_factors, trade_dates, hl_dict
        )

        X_tr = df_train_feat[feature_cols_template].values
        y_tr = df_train_feat["target"].values
        X_te = df_test_feat[feature_cols_template].values
        y_te = df_test_feat["target"].values

        # ④ 训练模型
        model = XGBClassifier(
            max_depth=5, n_estimators=200, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_tr, y_tr)

        # ⑤ 评估
        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        auc_scores.append(auc)
        print(f"Fold {fold+1}: AUC = {auc:.4f}")

    print("\n============================")
    print(f"Walk-forward mean AUC = {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    
# # ✅ 绘制 ROC 曲线
# fpr, tpr, _ = roc_curve(y, y_prob)
# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, label='ROC Curve')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

# # ✅ 绘制混淆矩阵
# cm = confusion_matrix(y, y_pred)
# plt.figure(figsize=(4, 4))
# plt.imshow(cm, cmap='Blues')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.xticks([0, 1], ['Down', 'Up'])
# plt.yticks([0, 1], ['Down', 'Up'])
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
# plt.grid(False)
# plt.show()

###批量画图结束,尝试下曲面

# half_life_range = range(2, 21)

# plot_df = pd.DataFrame(index = df.index)
# for factor in factors:
#     result = []
#     factor_data = macro_factors[macro_factors['类型'] == factor]
#     factor_data.index = [trade_dates[trade_dates <= d][-1] for d in factor_data.index]
#     for hl in half_life_range:
#         λ = np.log(2) / hl
#         plot_df[f'{factor}_decay'] = np.nan
#         # 构造指数衰减因子（仍用 max 填充逻辑）
#         for date, value in factor_data['今值'].dropna().items():
#             start_idx = plot_df.index.get_loc(date)
#             for i in range(20):
#                 if start_idx + i >= len(df):
#                     break
#                 t = plot_df.index[start_idx + i]
#                 decay_value = value * np.exp(-λ * i)
#                 if pd.isna(plot_df.at[t, f'{factor}_decay']):
#                     plot_df.at[t, f'{factor}_decay'] = decay_value
#                 else:
#                     plot_df.at[t, f'{factor}_decay'] = max(plot_df.at[t, f'{factor}_decay'], decay_value)        
#         for lag in range(1,max_lag+1):
#             y = df['ret'].shift(-1).rolling(lag).apply(lambda x: np.prod(1 + x) - 1, raw=True)
#             x = plot_df[f'{factor}_decay']
#             valid = x.notna() & y.notna()
#             ic = spearmanr(x[valid], y[valid])[0] if valid.sum() > 0 else np.nan
#             result.append({'half_life':hl,'lag':lag,'IC':ic})

#     df_long = pd.DataFrame(result)
#     df_pivot = df_long.pivot(index='half_life', columns='lag', values='IC')
            
    
    # pio.renderers.default='browser'
    
    # # df_pivot: index = half_life, columns = lag, values = IC
    # X = df_pivot.columns.values.astype(str)  # lag，转成 str 以便 hover 显示更好看
    # Y = df_pivot.index.values.astype(str)    # half-life
    # Z = df_pivot.values                      # IC 值矩阵
    
    # # 构建 Plotly 曲面图
    # fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    # # 设置布局
    # fig.update_layout(
    #     title=f'{factor} IC值曲面图（滞后窗口 × 半衰期）',
    #     scene=dict(
    #         xaxis_title='未来累计收益窗口（滞后天数）',
    #         yaxis_title='半衰期（交易日）',
    #         zaxis_title='Spearman IC',
    #     ),
    #     autosize=True,
    # )
    
    # # 展示交互图
    # fig.show()

###曲面结束
# half_life_range = range(2, 21)
# ic_at_10 = []

# for hl in half_life_range:
#     λ = np.log(2) / hl
#     df['cpi_decay'] = np.nan

#     # 构造指数衰减因子（仍用 max 填充逻辑）
#     for date, value in df['CPI'].dropna().items():
#         start_idx = df.index.get_loc(date)
#         for i in range(20):
#             if start_idx + i >= len(df):
#                 break
#             t = df.index[start_idx + i]
#             decay_value = value * np.exp(-λ * i)
#             if pd.isna(df.at[t, 'cpi_decay']):
#                 df.at[t, 'cpi_decay'] = decay_value
#             else:
#                 df.at[t, 'cpi_decay'] = max(df.at[t, 'cpi_decay'], decay_value)

#     # 计算未来10日 simple return
#     y = df['ret'].shift(-1).rolling(10).apply(lambda x: np.prod(1 + x) - 1, raw=True)
#     x = df['cpi_decay']
#     valid = x.notna() & y.notna()
#     ic = spearmanr(x[valid], y[valid])[0] if valid.sum() > 0 else np.nan
#     ic_at_10.append(ic)

# # 可视化
# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Kaiti SC'
# plt.rcParams['axes.unicode_minus'] = False

# plt.plot(half_life_range, ic_at_10, marker='o')
# plt.title('不同半衰期下的滞后10日IC')
# plt.xlabel('half-life（交易日）')
# plt.ylabel('第10日累计收益IC')
# plt.grid(True)
# plt.show()



