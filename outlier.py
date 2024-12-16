# import pandas as pd
# import numpy as np
# from scipy.stats import chi2
# from numpy.linalg import inv, det
# from scipy.spatial.distance import mahalanobis
#
# # 加载数据
# # file_path = 'F://光谱数据//模型训练数据//奥普天成 苹果数据//苹果光谱数据_分类276.csv'
# file_path ='F:\光谱数据\kiwi整合数据 ——谱研   暑假期间检测//kiwi-1-分类.csv'
# data = pd.read_csv(file_path)
#
# # 假设CSV文件的每一列都是一个特征，并且没有标签列
# X = data.values
# def mahalanobis_distance(x, data):
#     """计算x到数据集data的Mahalanobis距离。"""
#     mean = np.mean(data, axis=0)
#     cov_matrix = np.cov(data, rowvar=False)
#     inv_cov_matrix = inv(cov_matrix)
#     diff = x - mean
#     md = mahalanobis(diff, mean, inv_cov_matrix)
#     return md
#
# # 计算所有点的Mahalanobis距离
# mds = np.apply_along_axis(mahalanobis_distance, 1, X, X)
# # 使用卡方分布的临界值作为阈值
# p = X.shape[1]  # 特征数量
# threshold = chi2.ppf((1 - 0.001), df=p)  # 显著性水平为0.01
#
# # 识别异常值
# outliers = np.where(mds > threshold)
# # 打印异常值的索引
# print("异常值的索引：", outliers[0])
#
# # 打印异常值的数据
# print("异常值的数据：")
# print(data.iloc[outliers[0], :])
#
# # 移除异常值
# data_cleaned = data.drop(index=outliers[0])
# processed_file_path = 'F://光谱数据//模型训练数据//奥普天成 苹果数据//cleaned_苹果光谱数据_分类276.csv'
# data_cleaned.to_csv(processed_file_path, index=False)

