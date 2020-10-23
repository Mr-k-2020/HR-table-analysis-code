'''第6章02分类'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pydotplus
import os
os.environ["PATH"]+=os.pathsep + "E:/package_download/bin/"     #graphviz需要添加环境变量

#sl:satisfaction_level---Fasle:MinMaxScaler;True:StandardScaler
#le:last_evaluation---False:MinMaxScaler;True:Standardscaler
#npr:number_project---False:MinMaxScaler;True:StandardScaler
#amh:average_monthly_hours--False:MinMaxScaler;True:StandardScaler
#tsc:time_spend_company--False:MinMaxScaler;True:StandardScaler
#wa:Work_accident--False:MinMaxScaler;True:StandardScaler
#pl5:promotion_last_5years--False:MinMaxScaler;True:StandardScaler
#dp:department--False:LabelEncoding;True:OneHotEncoding
#slr:salary--False:LabelEncoding;True:OneHotEncoding
def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl5=False, dp=False, slr=False, lower_d=False, ld_n=1):
    df = pd.read_csv(r"E:\python_code\pycharm_code\coding-185\data\HR.csv")

    #1 清洗数据 去除异常值
    df = df.dropna(subset=["satisfaction_level", "last_evaluation"])        #指定要处理的列
    df = df[df["satisfaction_level"]<=1][df["salary"]!="nme"]
    #2 得到标注
    label = df["left"]
    df = df.drop("left", axis=1)
    #3 特征选择
    #4 特征处理
    scaler_lst = [sl, le, npr, amh, tsc, wa, pl5]       #获取函数的参数列表，未定义参数时传入默认值
    column_lst = ["satisfaction_level", "last_evaluation", "number_project",
                  "average_monthly_hours","time_spend_company","Work_accident",
                  "promotion_last_5years"]
    for i in range(len(scaler_lst)):
        #MinMaxScaler().fit_transform()或StandardScaler().fit_transform()需要传入格式为一行数据
        #数据输入前需要先reshape(-1,1)转成一行，返回前再转换为一列数据reshape(1,-1)转成一列再输出
        if not scaler_lst[i]:
            # 归一化0-1
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            # 标准化均值为0方差为1
            df[column_lst[i]] = StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1,-1)[0]
    scaler_lst = [slr, dp]
    column_lst = ["salary", "department"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            # 直接编码
            if column_lst[i] == "salary":
                #对salary直接赋值low:0,medium:1,high:2，若不赋值设置，则salary列会按照字母顺序，high:0,medium:1,low:2
                df[column_lst[i]] = [map_salary(s) for s in df["salary"].values]
            else:
                #对department0-6的值归一化
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            # 转成one-hot编码
            df = pd.get_dummies(df, columns = [column_lst[i]])
    if lower_d:
        return PCA(n_components = ld_n).fit_transform(df.values), label
    return df, label

#设置字典，分别存low、medium、high的编码值
d = dict([("low", 0), ("medium", 1), ("high", 2)])
def map_salary(s):          #仅获得字典值
    return d.get(s, 0)

def hr_modeling(features, label):
    #划分数据集，训练集、验证集、测试集
    from sklearn.model_selection import train_test_split
    f_v = features.values               #获取特征值
    f_n = features.columns.values       #获取特征名称
    l_v = label.values                  #获取标注值
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
    print(len(X_train), len(X_validation), len(X_test))

    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.externals.six import StringIO
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import SGD
    from sklearn.ensemble import GradientBoostingClassifier

    mdl = Sequential()
    mdl.add(Dense(50, input_dim=len(f_v[0])))
    mdl.add(Activation("sigmoid"))
    mdl.add(Dense(2))
    mdl.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    mdl.compile(loss="mean_squared_error", optimizer="adam")
    mdl.fit(X_train, np.array([[0,1] if i==1 else [1, 0] for i in Y_train]), nb_epoch=50, batch_size=500)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    f = plt.figure()
    data_lst = ["Train", "Validation", "Test"]
    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = mdl.predict_classes(X_part)        #模型预测
        f.add_subplot(1, 3, i+1)
        fpr, tpr, threshold = roc_curve(Y_part, Y_pred)
        plt.plot(fpr, tpr)
        print("NN", "AUC", auc(fpr, tpr))
        print("NN", "AUC_Score", roc_auc_score(Y_part, Y_pred))
        print(data_lst[i])                  #输出模型名称和准确率等
        print("NN", "-ACC:", accuracy_score(Y_part, Y_pred))
        print("NN", "-REC:", recall_score(Y_part, Y_pred))
        print("NN", "-F1:", f1_score(Y_part, Y_pred))
    plt.show()
    return

    models = []         #创建模型列表，以元组形式存储，用于存模型名称和模型
    models.append(("KNN", KNeighborsClassifier(n_neighbors=3)))         #允许的临近样本个数
    models.append(("GaussianNB", GaussianNB()))
    models.append(("BernoulliNB", BernoulliNB()))
    models.append(("DecisionTreeGini", DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))
    models.append(("SVM", SVC(C=10)))        #参数C设置类别错分的惩罚度C越大惩罚越大，C越大计算时间越长
    models.append(("RandomForst", RandomForestClassifier(max_features=None, bootstrap=True, n_estimators=11)))
    models.append(("Adaboost", AdaBoostClassifier()))
    models.append(("LogisticRegression", LogisticRegression()))
    #max_features采用全量特征，bootstrap有放回取样本，建立的决策树数量
    models.append(("GDBT", GradientBoostingClassifier(max_depth=6, n_estimators=100)))

    for clf_name, clf in models:
        clf.fit(X_train, Y_train)           #拟合模型
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        data_lst = ["Train", "Validation", "Test"]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)        #模型预测
            print(data_lst[i])                  #输出模型名称和准确率等
            print(clf_name, "-ACC:", accuracy_score(Y_part, Y_pred))
            print(clf_name, "-REC:", recall_score(Y_part, Y_pred))
            print(clf_name, "-F1:", f1_score(Y_part, Y_pred))
            # dot_data = StringIO()
            # export_graphviz(clf, out_file = dot_data, feature_names = f_n, class_names=["NL", "L"],
            #                 filled=True, rounded=True, special_characters=True)
            # #feature_names显示的特征名称，class_name为标注名称，filled、rounded、special_characters用于美观的可视化
            # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            # graph.write_pdf("dt_tree.pdf")



def main():
    features, label = hr_preprocessing(sl=True)
    hr_modeling(features, label)

if __name__=="__main__":
    main()

