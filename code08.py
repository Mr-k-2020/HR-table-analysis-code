'''HR表数据预处理'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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



def main():
    print(hr_preprocessing(sl=True, le=True))

if __name__=="__main__":
    main()

