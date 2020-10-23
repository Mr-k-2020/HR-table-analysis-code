'''半监督分类'''
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()         #获取内置的iris数据
labels = np.copy(iris.target)       #iris是对象类型，将标记复制出来
random_unlabeled_points = np.random.rand(len(iris.target))       #根据样本数建立随机数
random_unlabeled_points = random_unlabeled_points < 0.7          #<0.7的置true,否则置false
Y = labels[random_unlabeled_points]              #将True的标注置为Y,Y存无标注的数据的真实标注
labels[random_unlabeled_points] = -1             #对选取得无标注数据置-1
print("Unlabeled Number:", list(labels).count(-1))      #统计置-1的个数,即无标签的个数

from sklearn.semi_supervised import LabelPropagation
label_prop_model = LabelPropagation()                   #引入无监督的包
label_prop_model.fit(iris.data, labels)                 #拟合有标注和无标注的数据
Y_pred = label_prop_model.predict(iris.data)            #对data预测，结果输出Y_pred
Y_pred = Y_pred[random_unlabeled_points]                #仅取无标注样本的预测标注
from sklearn.metrics import accuracy_score, recall_score, f1_score
#计算预测标注和真实标注的准确率
print("ACC:", accuracy_score(Y, Y_pred))
print("REC:", recall_score(Y, Y_pred, average="micro"))     #计算召回率和F1值时，面对多分类模型，需要用参数micro
print("F-Score", f1_score(Y, Y_pred, average="micro"))

