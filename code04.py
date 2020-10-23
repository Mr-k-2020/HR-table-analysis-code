'''分组分析'''
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

'''交叉分析'''
pd.set_option('display.max_columns', None)      #显示完整的列
pd.set_option('display.max_rows', None)         #显示完整的行
df = pd.read_csv(r"E:\python_code\pycharm_code\coding-185\data\HR-1.csv")

sns.set_context(font_scale=1.5)
#离散值分组
# sns.barplot(x="salary", y="left", hue="department", data=df)
# plt.show()
#对连续值分组，用于之后的分析
sl_s = df["satisfaction_level"]
#大小排列sl_s的值，按顺序绘制
# sns.barplot(list(range(len(sl_s))), sl_s.sort_values())
# plt.show()

'''相关分析'''
'''连续属性相关性分析'''
#两两分析相关性，颜色蓝正相关越大，越红负相关
#vmax, vmin设置上下界
# sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap=sns.color_palette("RdBu", n_colors=128))
# plt.show()

'''离散属性相关性分析'''
#2类或多类离散属性直接转为0和1或者012等，可以直接使用Pearson相关系数，但存在失真
#离散数据采用熵计算相关性
s1 = pd.Series(["X1", "X1", "X2", "X2", "X2", "X2"])
s2 = pd.Series(["Y1", "Y1", "Y1", "Y2", "Y2", "Y2"])
#定义计算信息熵函数
def getEntropy(s):
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = s.groupby(by=s).count().values/float(len(s))      #返回列表，每类分布概率的列表
    return -(np.log2(prt_ary)*prt_ary).sum()                    #对分布概率列表计算熵，求和
print("Entropy:", getEntropy(s1))

#定义条件熵函数
def getCondEntropy(s1, s2):
    #构造结构体，用于更好的表示类之间的关系
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
        # print(d)
        #结构体构造结果：{'X1': ['Y1', 'Y1'], 'X2': ['Y1', 'Y2', 'Y2', 'Y2']}
    return sum([getEntropy(d[k])*len(d[k])/float(len(s1)) for k in d])  #k为d的索引
print("CondEntropy:", getCondEntropy(s1, s2))

#定义熵增溢函数
def getEntropyGain(s1, s2):
    return getEntropy(s2) - getCondEntropy(s1, s2)
print("EntropyGain", getCondEntropy(s1, s2))

#定义熵增益率函数不具有对称性,s1s2的增益率函数值与s2s1的增益率函数值不同
def getEntropyGainRatio(s1, s2):
    return getEntropyGain(s1, s2)/getEntropy(s2)
print("EntropyGainRatio", getEntropyGainRatio(s1, s2))

#离散型数据相关性度量：离散值相关性
import math
def getDiscreteCorr(s1, s2):
    return getEntropyGain(s1, s2)/math.sqrt(getEntropy(s1)*getEntropy(s2))
print("DiscreteCorr", getDiscreteCorr(s1, s2))

#使用基尼系数分组数据
def getProbSS(s):           #定义概率平方和函数
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = s.groupby(by=s).count().values/float(len(s))      #或 s.value_counts()/float(len(s)) 亦可实现计算离散数据分布
    return sum(prt_ary**2)
def getGini(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return 1-sum([getProbSS(d[k])*len(d[k])/float(len(s1)) for k in d])
print("Gini", getGini(s1, s2))

'''因子分析(成分分析)'''
from sklearn.decomposition import PCA
my_pca = PCA(n_components=7)
lower_mat = my_pca.fit_transform(df.drop(labels=["salary", "department", "left"], axis=1))
print("Ratio", my_pca.explained_variance_ratio_)        #重要性占比
#.corr()计算两两之间的相关系数并画图
sns.heatmap(pd.DataFrame(lower_mat).corr(), vmin=-1, vmax=1, cmap=sns.color_palette("RdBu", n_colors=128))
#7种属性之间相关性为0几乎不相关
plt.show()