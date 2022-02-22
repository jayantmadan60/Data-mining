import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from sklearn import preprocessing
from sklearn.decomposition import PCA

# load the dataset(sl = sepal length, sw = sepal width, pl = petal length, pw = petal width,class = Target)
df = pd.read_csv('C:\\Users\\Authorized User1\\Desktop\\Data mining\\assignment 2\\iris.txt',
                 names=["sl","sw","pl","pw","class"])
print(df.info())
print (df.head())

# 2D scatter plots of the four features
sns.set_theme(style="darkgrid")
sns.pairplot(df,hue='class',diag_kind="kde",markers=['o','v','s'])
plt.show()

# 3D scatter plot of three features: sepal length, sepal width, petal width.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['sl'], df['sw'], df['pw'], color = 'green',s = 20,depthshade=True,marker='o')
plt.title("3D scatter plot")
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Width')
plt.show()

# Visualization of the feature matrix (column 1-4) as an image.
dfd = df.drop('class',axis=1)
plt.imshow(np.asmatrix(dfd), aspect = 'auto')
plt.show()

# For each class, generate histograms for the four features.
fig, ax = plt.subplots(2,2)
sns.histplot(ax=ax[0,0],data= df,x='sl',hue='class',bins= 20,alpha = 0.3)
sns.histplot(ax=ax[0,1],data= df,x='sw',hue='class',bins= 20,alpha = 0.3)
sns.histplot(ax=ax[1,0],data= df,x='pl',hue='class',bins= 20,alpha = 0.3)
sns.histplot(ax=ax[1,1],data= df,x='pw',hue='class',bins= 20,alpha = 0.3)
plt.show()

#For each class, generate boxplots of the four features.
fig, ax = plt.subplots(2,2)
sns.boxplot(ax=ax[0,0],data= df,x = 'class',y='sl')
sns.boxplot(ax=ax[0,1],data= df,y='sw',x='class')
sns.boxplot(ax=ax[1,0],data= df,y='pl',x='class')
sns.boxplot(ax=ax[1,1],data= df,y='pw',x='class')
plt.show()

#Calculate the correlation matrix of the four features
print (df.corr())

#Visualize the correlation matrix as an image.
sns.heatmap(df.corr())
plt.show()

#Create a parallel coordinate plot of the four features.
plt.style.context(("ggplot", "seaborn"))
pd.plotting.parallel_coordinates(df, 'class', color=('red', 'blue', 'black'),alpha = 0.5)
plt.title("parallel coordinate plot")
plt.legend()
plt.show()




#Make a function for Minkowski Distance.
def minkowski(x,y,r):
    a = []
    for (i, j, k, l) in zip(df[x[0]], df[x[1]], df[x[2]], df[x[3]]):
        a.append((((abs(i - y[0])) ** r) + ((abs(j - y[1])) ** r) + ((abs(k - y[2])) ** r) + ((abs(l - y[3])) ** r)) ** ( 1 / r))
    return a




#Make a function for T-statistics Distance.
dt = pd.read_csv('C:\\Users\\Authorized User1\\Desktop\\Data mining\\assignment 2\\hw2_ts.txt',
                 names=["c1","c2"])
def ttest(d1, d2):
    return ((np.mean(d1) - np.mean(d2)) / (np.sqrt((sem(d1))**2 + sem(d2)**2)))




#Make a function for Mahalanobis Distance
def mahalnobis(x,y,m):
    a = []
    for (i, j, k, l) in zip(df[x[0]], df[x[1]], df[x[2]], df[x[3]]):
        x = np.array([i, j, k, l])
        y = np.array(y)
        v = np.array([(x[0] - y[0]), (x[1] - y[1]), (x[2] - y[2]), (x[3] - y[3])])
        a.append(np.sqrt(np.dot(np.dot(v, m), v.transpose())))
    return a

#Make a function for Minkowski Distance and plot.
def minkowski(x,y,r):
    a = []
    for (i, j, k, l) in zip(df[x[0]], df[x[1]], df[x[2]], df[x[3]]):
        a.append((((abs(i - y[0])) ** r) + ((abs(j - y[1])) ** r) + ((abs(k - y[2])) ** r) + ((abs(l - y[3])) ** r)) ** ( 1 / r))
    return a

q = [5.0000, 3.5000, 1.4600, 0.2540]
p= ['sl','sw','pl','pw']
# r = 1
sns.scatterplot(range(len(minkowski(p,q,1))),minkowski(p,q,1),hue=df['class'])
plt.xlabel('Number')
plt.ylabel('Minkowski distance for (r = 1)')
plt.show()
# r = 2
sns.scatterplot(range(len(minkowski(p,q,2))),minkowski(p,q,2),hue=df['class'])
plt.xlabel('Number')
plt.ylabel('Minkowski distance for (r = 2)')
plt.show()
# r = 10
sns.scatterplot(range(len(minkowski(p,q,10))),minkowski(p,q,10),hue=df['class'])
plt.xlabel('Number')
plt.ylabel('Minkowski distance for (r = 10)')
plt.show()


#Calculate Mahalanobis distances and plot
r = np.linalg.inv(np.cov(np.matrix([df['sl'],df['sw'],df['pl'],df['pw']])))
p = ['sl','sw','pl','pw']
q = [5.0000, 3.5000, 1.4600, 0.2540]

sns.scatterplot(range(len(mahalnobis(p,q,r))),(mahalnobis(p,q,r)),hue=df['class'])
plt.xlabel('Number')
plt.ylabel('Mahalonobis distance')
plt.show()

#Normalize the feature matrix of the IRIS dataset
normalize = df.drop('class',axis=1)
normalize = (((normalize-(np.mean(normalize)))/np.std(normalize)))
print (normalize)

#Calculate the correlation matrix
print (normalize.corr())

# load the dataset
df = pd.read_csv('C:\\Users\\Authorized User1\\Desktop\\Data mining\\assignment 2\\iris.txt',
                 names=["sl","sw","pl","pw","class"])
dr = df.drop('class',axis=1)
pca = PCA(n_components=4)
dr_fit = pca.fit(dr)
dr_transform = pca.transform(dr)
dr_df = pd.DataFrame(dr_transform, columns = ['p1','p2','p3','p4'])

#Create 2D scatter plots of each pair of the four components
sns.pairplot(dr_df)
plt.show()

#3D scatter plot of the first three components
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dr_df['p1'], dr_df['p2'], dr_df['p3'], color = 'green',s = 20,depthshade=True,marker='v',)
plt.title("3D scatter plot")
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('p3')
plt.show()

#Obtain the variance of each component and visualize in a figure plot.
print(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#Calculate the correlation matrix of the four components
print (dr_df.corr())

#load the dataset
dt = pd.read_csv('C:\\Users\\Authorized User1\\Desktop\\Data mining\\assignment 2\\hw2_ts.txt',names=["c1","c2"])
print (dt.head())

#Visualize the two time series in one figure plot.
dt.plot()
plt.xlabel('index of data points')
plt.ylabel('Time series data')
plt.show()

#Calculate the T-statistics distance
print (ttest(dt['c1'],dt['c2']))

#Calculate the correlation of the two time series
print (dt.corr())

