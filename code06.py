'''特征选择'''
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel

#造数据，ABC是属性D是标签
df = pd.DataFrame({"A":ss.norm.rvs(size=10),
                   "B":ss.norm.rvs(size=10),
                   "C":ss.norm.rvs(size=10),
                   "D":np.random.randint(low=0, high=2, size=10)})        #low可取到high取不到，从0，1种选择10次
print("data\n", df)

X = df.loc[:, ["A", "B", "C"]]
Y = df.loc[:, "D"]
#使用过滤思想
skb = SelectKBest(k=2)
skb.fit(X,Y)
print("skb\n", skb.transform(X))     #这种方法保留AB

#使用包裹思想
rfe = RFE(estimator=SVR(kernel="linear"), n_features_to_select=2, step=1)
print("rfe\n", rfe.fit_transform(X, Y))      #保留AC

#使用嵌入思想
sfm = SelectFromModel(estimator=DecisionTreeRegressor(), threshold=0.1)
print("sfm\n", sfm.fit_transform(X, Y))
