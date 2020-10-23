import pandas as pd
import numpy as np
import scipy.stats as ss

# pd.set_option('display.max_columns', None)      #显示完整的列
# pd.set_option('display.max_rows', None)         #显示完整的行
# df = pd.read_csv(r"E:\python_code\pycharm_code\coding-185\data\HR.csv")

#正态分布假设检验
norm_dist = ss.norm.rvs(size=20)         #造符合正态分布的数据
print("正态分布数据：",norm_dist)         #打印数据
'''正态分布检验，返回两个值，statistic统计值，pvalue显著性值'''
print("正态分布检验结果",ss.normaltest(norm_dist))             #显著性<0.05则接受检验
print("*********************************************")
#卡方分布检验
'''卡方检验返回4个结果：检验统计量、P值、自由度、理论分布（平均分布）'''
print("卡方分布检验结果", ss.chi2_contingency([[15, 95], [85, 5]]))         #P值越小卡方值越大，数据间偏差程度越大
print("*********************************************")
#独立t分布检验
'''返回两个数：检验统计量、P值，同一分布下数据量越大，数据间差别越小'''
print("T分布检验")      #t分布是小样本趋于正态分布
print(ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(size=20)))             #P值小
print(ss.ttest_ind(ss.norm.rvs(size=100), ss.norm.rvs(size=200)))           #P值大
print("*********************************************")

#方差检验
'''返回两个数：检验统计量，P值'''
print("F分布检验")
print(ss.f_oneway([49, 50, 39, 40, 43], [28, 32, 30, 26, 34], [38, 40, 45, 42, 48]))    #P值越小均值相关越小

#qq图用于验证数据是否为正态分布
# from statsmodels.graphics.api import qqplot
# from matplotlib import pyplot as plt
# qqplot(ss.norm.rvs(size=100))
# plt.show()     #数据越接近左下到右上的对角线，则越接近正态分布

#计算相关系数
s1 = pd.Series([0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5])
s2 = pd.Series([0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1])
'''Series之间计算相关系数'''
#.corr()不设置参数则对表中所有列两两计算相关系数，设置参数对指定两列计算相关系数
print("计算相关系数：\n", s1.corr(s2))
print("spearman相关系数:\n", s1.corr(s2, method="spearman"))
'''DataFrame内计算相关系数,计算列之间的相关系数'''
df = pd.DataFrame(np.array([s1, s2]).T)
print("dataframe相关系数\n", df.corr())
print("dataframe spearman相关系数\n", df.corr(method="spearman"))

'''线性回归'''
x = np.arange(10).astype(np.float).reshape((10, 1))
y = x * 3 + 4 + np.random.random((10, 1))
print("x:", x)
print("y:", y)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
res = reg.fit(x, y)
y_pred = reg.predict(x)
print("参数：", reg.coef_)
print("截距：", reg.intercept_)

'''数据降维，主成分分析'''
data = np.array([np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]),
                 np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])]).T

from sklearn.decomposition import PCA   #sklearn的PCA降维本质是奇异值分解，非PCA降维方法
lower_dim = PCA(n_components=1)         #降至一维
print("降维后的维度", lower_dim.fit(data))
print("降维后信息保存量", lower_dim.explained_variance_ratio_)
print("SVD结果", lower_dim.fit_transform(data))

#手写PCA
from scipy import linalg
def myPCA(data, n_components=10000):        #最多可以进行10000维数据的降维
    mean_vals = np.mean(data, axis=0)
    mid = data - mean_vals
    cov_mat = np.cov(mid, rowvar=False)     #rowvar=False用于不针对行进行协方差计算
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    # print(eig_vals)
    eig_vals_index = np.argsort(eig_vals)[:-(n_components+1):-1]        #从小到大排序，取最后1个作为特征值[::-1]用于从后取
    # print(np.argsort(eig_vals))
    # print(eig_vals_index)
    eig_vects = eig_vects[:, eig_vals_index]
    low_dim_mat = np.dot(mid, eig_vects)
    return low_dim_mat, eig_vals

print("PCA结果", myPCA(data, n_components=1))