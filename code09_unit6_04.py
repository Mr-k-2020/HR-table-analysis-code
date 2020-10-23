'''无监督学习'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#造1000个数据
n_samples = 1000
circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
moons = make_moons(n_samples=n_samples, noise=0.05)
blobs = make_blobs(n_samples=n_samples, random_state=8, center_box=(-1,1), cluster_std=0.1)      #center_box确定上下界，cluster_std确定类的标准差
random_data = np.random.rand(n_samples, 2), None        #由于circles、moons、blobs产生的数据集都自带标签，所以随机数据使用None占据标签的位置

colors = "bgrcmyk"              #设置分类的颜色
data = [circles, moons, blobs, random_data]
models=[("None",None),("Kmeans",KMeans(n_clusters=3)),
        ("DBSCAN",DBSCAN(min_samples=3,eps=0.2)),           #参数为最小数和最小邻域值
        ("Agglomerative",AgglomerativeClustering(n_clusters=3,linkage="ward"))]
from sklearn.metrics import silhouette_score
f=plt.figure()
for inx, clt in enumerate(models):      #enumerate()返回元素索引和元素值
    clt_name, clt_entity=clt            #clt包含模型名称和模型
    for i, dataset in enumerate(data):
        X, Y = dataset                  #X为造的数据，Y为自带的标注
        if not clt_entity:              #如果定义的方法为空，则执行
            clt_res = [0 for item in range(len(X))]         #设置数据类型全为0
        else:
            clt_entity.fit(X)           #拟合模型
            clt_res = clt_entity.labels_.astype(np.int)     #获得数据类型，.astype(np.int)转为int型
        f.add_subplot(len(models), len(data), inx*len(data)+i+1)        #设置画布位置
        plt.title(clt_name)
        try:
            print(clt_name,i,silhouette_score(X,clt_res))
        except:         #遇到None时跳过
            pass
        [plt.scatter(X[p,0], X[p,1], color=colors[clt_res[p]]) for p in range(len(X))]      #根据类别选择颜色
plt.show()
