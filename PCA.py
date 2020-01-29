# 1 make up some data for pca

# 2 PCA fucntion for Sklearn

# 3 plot the data

import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# sample data set
genes=['gene'+ str(i) for i in range(1,101)]

wt=['wt'+ str(i) for i in range(1,6)]
ko=['ko'+ str(i) for i in range(1,6)]
# pandas to store fake data
data=pd.DataFrame(columns=[*wt,*ko],index=genes)

for gene in data.index:
    data.loc[gene, 'wt1':'wt5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
    data.loc[gene, 'ko1':'ko5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)

print(data.head())
print(data.shape)
# first center and scale the data

scaled_data=preprocessing.scale(data.T)

pca=PCA()

pca.fit(scaled_data)

pca_data=pca.transform(scaled_data)

# scree plot to check how many PCA should be made
#percentage of variation
per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
labels=['PC'+ str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.xlabel('percentage')
plt.ylabel('Principal component')
plt.title('SCREE PLOT')
plt.show()

pca_df = pd.DataFrame(pca_data, index=[*wt,*ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("MY PCA")

plt.xlabel('PC1- {0}'.format(per_var[0]))
plt.ylabel('PC2 - {0}'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample] , pca_df.PC2.loc[sample]))
plt.show()