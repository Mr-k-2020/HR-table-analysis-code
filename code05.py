'''异常值处理'''
import  pandas as pd
df = pd.DataFrame({"A":['a0', 'a1', 'a1', 'a2', 'a3', 'a4'],
                   "B":['b0', 'b1', 'b2', 'b2', 'b3', None],
                   "C":[1, 2, None, 3, 4, 5],
                   "D":[0.1, 10.2, 11.4, 8.9, 9.1, 12],
                   "E":[10, 19, 32, 25, 8, None],
                   "F":['f0', 'f1', 'g2', 'f3', 'f4', 'f5']})

print("isnull\n", df.isnull())          #True出现在空值位置上
print("dropna\n", df.dropna())          #删除所有空值所在行
print("dropBNan\n", df.dropna(subset=["B"]))        #删除B属性空值
print("duplicatedA\n", df.duplicated(["A"]))        #标记A属性的重复值
print("duplicatedAB\n", df.duplicated(["A", "B"]))   #标记AB同时重复的行
print("drop_duplicatedA\n", df.drop_duplicates(["A"]))  #删除A重复值
#drop_duplicates()函数keep参数可为first保留首个，last保留最后一个，False全都不要；inplace为True或False意为是否修改原表
print(df.drop_duplicates(["A"], keep='first', inplace=False))
print("空值全填充\n", df["B"].fillna("b*", inplace=False))         #fillna()正对df全填充或针对Series填充
print("空值填充列均值\n", df["E"].fillna(df["E"].mean()))
#插值 默认情况下，首尾空填充临近值，中间空填充平均值
print("插值 仅能针对Series插值\n", df["E"].interpolate())
print("插值 指定插值方法\n", df["E"].interpolate(method="spline", order=3))      #指定 三次样条插值
upper_q = df["D"].quantile(0.75)
lower_q = df["D"].quantile(0.25)
q_int = upper_q - lower_q
k = 1.5
print(df[df["D"]>lower_q - k*q_int][df["D"]<upper_q + k*q_int])
print("处理F列异常值g2")
print(df[[True if item.startswith("f") else False for item in list(df["F"].values)]])
