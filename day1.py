import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from scipy.stats import zscore

df=pd.read_csv("Titanic-Dataset.csv")

print("first five rown:\n",df.head())
print("basic info about data:\n",df.info())
print("data types of the columns:\n",df.dtypes)

print("\nnull values before imputation:\n",df.isnull().sum())
si1=SimpleImputer(missing_values=np.nan,strategy="mean")
si2=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
df["Age"]=si1.fit_transform(df[["Age"]])
df["Cabin"]=si2.fit_transform(df[["Cabin"]]).ravel()
df["Embarked"]=si2.fit_transform(df[["Embarked"]]).ravel()
print("null values after imputation:",df.isnull().sum())

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
print("after encoding:",df.head())

ss=StandardScaler()
df[["Age","Fare"]]=ss.fit_transform(df[["Age","Fare"]])
print("after scaling:",df.head())

plt.boxplot(df[['Age','Fare']])
plt.xticks([1,2],['Age','Fare'])
plt.title("Boxplot with outliers")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()

z_age = zscore(df["Age"])
z_fare = zscore(df["Fare"])

df = df[(abs(z_age) < 3) & (abs(z_fare) < 3)]

print("Number of outliers in fare with zscore more than 3:", (abs(z_fare)>3).sum())
print("Number of outliers in age with zscore more than 3:", (abs(z_age)>3).sum())
zfare=zscore(df["Fare"])
df=df[abs(zfare)<3]
plt.boxplot(df[['Age','Fare']])
plt.xticks([1,2],['Age','Fare'])
plt.title("Boxplot without outliers")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()