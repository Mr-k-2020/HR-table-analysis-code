import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)      #显示完整的列
pd.set_option('display.max_rows', None)         #显示完整的行

df = pd.read_csv(r"E:\python_code\pycharm_code\coding-185\data\HR.csv")
# print(df.head(10))

print("satisfaction_level")
'''第1步 先看看有没有异常值'''
sl_s = df["satisfaction_level"]
'''异常值分析'''
print(sl_s[sl_s.isnull()])          #.isnull() 判断是否为空，非空False为空True
# sl_s.dropna()           #丢弃空值
# sl_s.fullna()           #填充空值
print("avg", sl_s.mean())
print("std", sl_s.std())
print("max", sl_s.max())
print("min", sl_s.min())
print("median", sl_s.median())
print("25%", sl_s.quantile(q=0.25))
print("75%", sl_s.quantile(q=0.75))
'''分布分析'''
print("skew", sl_s.skew())
print("kurt", sl_s.kurt())

print("satisfaction_level", np.histogram(sl_s.values, bins=np.arange(0.0, 1.1, 0.1)))     #每0.1间隔的数量大小

print("last_evaluation")
le_s = df["last_evaluation"]
q_low = le_s.quantile(q=0.25)
q_high = le_s.quantile(q=0.75)
q_interval = q_high - q_low
k = 1.5
le_s = le_s[le_s < q_high + k * q_interval][le_s > q_low - k * q_interval]      #Series的并列[]表示并列条件

print("last_evaluation", np.histogram(le_s.values, bins=np.arange(0.0, 1.1, 0.1)))

print("number_project")
np_s = df["number_project"]
print("last_evaluation 计数", np_s.value_counts())           #计数每个元素出现个数
print("last_evaluation 比例", np_s.value_counts(normalize=True))

print("avgrage_month_hours")
amh_s = df["average_monthly_hours"]
print("np.hist bins=10", np.histogram(amh_s.values, bins=10))       #等分10份
'''np.histogram左闭右开'''
print("np.hist max min", np.histogram(amh_s.values, bins=np.arange(amh_s.min(), amh_s.max()+10, 10)))
'''value_counts左开右闭'''
print("value_count max min", amh_s.value_counts(bins=np.arange(amh_s.min(), amh_s.max()+10, 10)))

print("Department")
d_s = df["department"]
print("原", d_s.tail())
print("value_count 原", d_s.value_counts())
d_s_1 = d_s.where(d_s!="sale").fillna(value="S").copy()         #对异常值填充
print("现", d_s_1.tail(3))
print("value_count 现", d_s_1.value_counts())

#简单对比分析
print("**********简单对比分析**************")
df = df.dropna(axis=0, how="any")       #axis=0按行删除，how=any行中只要有一个空值就删除
df = df[df["last_evaluation"]<=1][df["salary"]!="nme"][df["department"]!="sale"]

print("聚合groupby")
print(df.groupby("department").mean())          #根据某属性值对全表聚合，数字类型的取平均，字符类型被抹除
print(df.loc[:, ["last_evaluation", "department"]].groupby("department").mean())            #根据属性对部分列聚合
'''聚合两列，并应用自定义函数
.mean()替换为.apply(lambda x:x.max()-x.min())
.apply(lambda x:...)   无函数名直接应用，格式为 lambda 参数：操作(参数)
'''
print(df.loc[:, ["average_monthly_hours", "department"]].groupby("department")["average_monthly_hours"].apply(lambda x:x.max()-x.min()))    #计算极差

'''画统计图'''
# 柱状图,每个柱的高度有意义，每个柱代表不同类
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_style(style="whitegrid")            #设置图表样式
# sns.set_context(context="poster", font_scale=0.8)           #设置字体和字号
# sns.set_palette(sns.color_palette("RdBu", n_colors=7))      #设置绘图的颜色或色系
# sns.countplot(x="salary", hue="department", data=df)        #sns.countplot()直接绘图，以salary为组，department为柱，数据为值
# '''不同薪资水平和不同职位的人数'''
# plt.show()
# plt.title("SALARY")                 #标题
# plt.xlabel("Salary")                #横轴名称
# plt.ylabel("Number")                #纵轴名称
# plt.axis([0, 3, 0, 10000])          #设置柱状图的显示范围，设置横轴0-3， 纵轴0-10000
# '''bar(A, S) A是列表，描述每个柱的位置，默认从0开始，value_counts()表示每类数值， A和S有对应关系'''
# plt.bar(np.arange(len(df["salary"].value_counts()))+0.5 , df["salary"].value_counts(), width=0.5)               #+0.5是为了移动柱子，以适应横轴0-3的显示范围
# '''设置每个柱的标签位置，类名索引就是柱名'''
# plt.xticks(np.arange(len(df["salary"].value_counts()))+0.5, df["salary"].value_counts().index)       #+0.5是为了移动标题位置，适应显示范围
# for x, y in zip(np.arange(len(df["salary"].value_counts()))+0.5, df["salary"].value_counts()):
#     plt.text(x,y, y, ha="center", va="bottom")
#
# plt.show()

# 直方图 每个柱的面积有意义
# f = plt.figure()
# f.add_subplot(1, 3, 1)
# sns.distplot(df["satisfaction_level"], bins=10)                 #显示直方图和分布图
# f.add_subplot(1, 3, 2)
# sns.distplot(df["last_evaluation"], bins=10, kde=False)         #仅显示直方图不显示分布图
# f.add_subplot(1, 3, 3)
# sns.distplot(df["average_monthly_hours"], bins=10, hist=False)  #不显示直方图仅显示分布图
# plt.show()

#箱线图 很好的表示数据分布，中位数，四分位数等
# '''赋值x，画横箱；赋值y，画纵箱'''
# '''saturation控制箱的颜色，whis作为k，控制上下界,默认whis=1.5'''
# sns.boxplot(y=df["time_spend_company"], saturation=0.75, whis=3)
# plt.show()

# 折线图
# sub_df = df.groupby("time_spend_company").mean()
# print(sub_df)
# # sns.pointplot(sub_df.index, sub_df["left"])           #这样画的折线图仅是点的连接
# sns.pointplot(x="time_spend_company", y="left", data=df)        #这样的语句需指定数据源，并且包含误差线，误差线的一半代表标准差的大小

# 饼图 只能用plt画饼图
lbs = df["department"].value_counts().index         #标签单独存列表
explodes = [0.1 if i == "sales" else 0 for i in lbs]            #根据lbs顺序决定explods列表，表示间隔
'''画图，explode确定间隔，labels为标签，colors可以调用sns的调色盘'''
plt.pie(df["department"].value_counts(normalize=True), explode=explodes, labels=lbs, colors=sns.color_palette("Reds"))
plt.show()

