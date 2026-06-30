import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

df=pd.read_csv("Titanic-Dataset.csv")
print("data before preprocessing\n",df.head())
si1=SimpleImputer(missing_values=np.nan,strategy="mean")
si2=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
df["Age"]=si1.fit_transform(df[["Age"]])
df["Cabin"]=si2.fit_transform(df[["Cabin"]]).ravel()
df["Embarked"]=si2.fit_transform(df[["Embarked"]]).ravel()

df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True)

ohe=OneHotEncoder(sparse_output=False,drop="first")
dfa=pd.DataFrame(ohe.fit_transform(df[['Sex']]))
ohe1=OneHotEncoder(sparse_output=False)
dfe=pd.DataFrame(ohe1.fit_transform(df[['Embarked']]))
df["Sex"]=dfa
df=pd.concat([df,dfe],axis=1)
df.rename(columns={0:'S',1:'C',2:'Q'})
df.drop('Embarked',axis=1,inplace=True)

z_age = zscore(df["Age"])
z_fare = zscore(df["Fare"])
df = df[(abs(z_age) < 3) & (abs(z_fare) < 3)]
(abs(z_fare)>3).sum(),(abs(z_age)>3).sum()
zfare=zscore(df["Fare"])
df=df[abs(zfare)<3]
print("data after preprocessing\n",df.head())
print("mean of age column:",df["Age"].mean())
print("mean of fare column:",df["Fare"].mean())
print("median of age column:",df["Age"].median())
print("median of fare column:",df["Fare"].median())
print("mode of age column:",df["Age"].mode())
print("mode of fare column:",df["Fare"].mode())
print("standard deviation of age column:",df["Age"].std())
print("standard deviation of fare column:",df["Fare"].std())
plt.hist(df["Age"])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram of Age")
plt.show()
plt.hist(df["Fare"])
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Histogram of Fare")
plt.show()
plt.boxplot(df["Age"])
plt.xlabel("Age")   
plt.title("Boxplot of Age")
plt.show()
plt.boxplot(df["Fare"])
plt.xlabel("Fare")
plt.title("Boxplot of Fare")
plt.show()
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
print("from the histplot we can say that there are more number of people in titanic are at the age of 30,so most of them were middle aged")
print("many people are having with fare in the range of 0-10")
print("from the box plot we can say that,there are many outliers in fare column that age column.and that too with high value")
print("from the correlation matrix we can say that,there are some more correlated feature like \n(survived,fare)\n(sibsp,parch)\n(fare,sibsp)")
print("as we see,the fare and survived are good enough to correlate we can say that most of the survived people were first class people ")