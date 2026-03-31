import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler


df=pd.read_csv('customers.csv')

print(df.head())

print(df.describe())

#since kmeans is a distance metric we need to be carefull of the data which is highly varrying so we need to scale or normalize it

scaler=StandardScaler()

sc_data=scaler.fit_transform(df)

sc_df=pd.DataFrame(sc_data)

print(sc_data)

print(sc_df.describe())

kmns=KMeans(n_clusters=2,init='k-means++')

kmns.fit(sc_df)

print(kmns.inertia_)




pred=kmns.predict(sc_df)

sc_df['cluster']=pred
print(df.head())

wcss=[]

for i in range(1,21):
    kmk=KMeans(n_clusters=i,init='k-means++')
    kmk.fit(sc_df)
    wcss.append(kmk.inertia_)

#plt.figure(figsize=(6,3))

plt.plot(range(1,21),wcss,"D")
plt.title("elbow curve")
plt.xlabel("k value")
plt.ylabel("inertia")
plt.show()
