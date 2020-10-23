'''数据预处理'''
# 数据离散化
import numpy as np
import  pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

lst = [6, 8, 10, 15, 16, 24, 25, 40, 67]
print(pd.qcut(lst, q=3, labels=["low", "medium", "high"]))          #等频分箱
print(pd.cut(lst, bins=3, labels=["low", "medium", "high"]))        #等宽分箱

#归一化
arr = np.array([1, 4, 10, 15, 21])
print(MinMaxScaler().fit_transform(arr.reshape(-1, 1)))             #reshape(-1,1)意为输出不指定行数，但指定列数为1

#标准化
arr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
print(StandardScaler().fit_transform(arr.reshape(-1, 1)))
arr = np.array([1, 0, 0, 0, 0, 0, 0, 0])
print(StandardScaler().fit_transform(arr.reshape(-1, 1)))

#标签化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print(LabelEncoder().fit_transform(np.array(["down", "up", "up", "down"]).reshape(-1, 1)))      #数据编码

lb_encoder = LabelEncoder()         #one-hot编码
lb_encoder = lb_encoder.fit(np.array(["Red", "Yellow", "Blue", "Green"]))
lb_tran_f = lb_encoder.transform(np.array(["Red", "Yellow", "Blue", "Green"]))
print(lb_tran_f)
oht_encoder = OneHotEncoder().fit(lb_tran_f.reshape(-1, 1))
print(oht_encoder.transform(lb_encoder.transform(np.array(["Yellow", "Blue", "Green", "Green", "Red"])).reshape(-1, 1)).toarray())

#LDA降维：投影转换数据，相同标签的向量距离尽可能小，不用标签距离尽可能大
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

print(LinearDiscriminantAnalysis(n_components=1).fit_transform(X, Y))     #n_components表示降至多少维的数据
#LDA降维同样可以作为分类器进行分类
clf = LinearDiscriminantAnalysis(n_components=1).fit(X, Y)
print(clf.predict([[0.8, 1]]))













