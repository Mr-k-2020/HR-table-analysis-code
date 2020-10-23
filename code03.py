import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

'''交叉分析'''
pd.set_option('display.max_columns', None)      #显示完整的列
pd.set_option('display.max_rows', None)         #显示完整的行
df = pd.read_csv(r"E:\python_code\pycharm_code\coding-185\data\HR-1.csv")

#left属性，各部门间离职率是否有关系，独立t检验
#按department分组
dp_indices = df.groupby(by="department").indices                #indices()返回字典，key为department类名，value为包含的行号
sales_values = df["left"].iloc[dp_indices["sales"]].values      #根据部门确定行号，通过行号定位确定值，获得sales的离职情况
technical_values = df["left"].iloc[dp_indices["technical"]].values   #同上
print(ss.ttest_ind(sales_values, technical_values)[1])          #比较sales和technical离职t分布，输出p值
dp_keys = list(dp_indices.keys())                               #获得部门列表
dp_t_mat = np.zeros([len(dp_keys), len(dp_keys)])               #建立矩阵
#两两对比部门间离职情况，通过热力图反应两两关系
for i in range(len(dp_keys)):
    for j in range(len(dp_keys)):
        p_value = ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values,\
                               df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]       #两两计算P值
        if p_value < 0.05:
            dp_t_mat[i][j] = -1
        else:
            dp_t_mat[i][j] = p_value            #p值填入矩阵
#P值越大，相关性越大，差异越小；P值越小，相关性越小，差异越大
sns.heatmap(dp_t_mat, xticklabels=dp_keys, yticklabels=dp_keys)     #画热力图，颜色越深差异越大，越浅差异越小
plt.show()


#通过透视表反应关系
#透视表的值是left，索引用"promotion_last_5years", "salary"，列类用"work_accident"，left聚合方式为mean
piv_tb = pd.pivot_table(df, values="left", index=["promotion_last_5years", "salary"],\
                                                  columns=["Work_accident"], aggfunc=np.mean)
sns.heatmap(piv_tb, vmin=0, vmax=1, cmap=sns.color_palette("Reds", n_colors=256))
plt.show()
