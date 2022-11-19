import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostClassifier

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format", lambda x:"%.3f" %x)
pd.set_option("display.width",500)

df=pd.read_csv("..") #Note:The dataset is private so It can't be shared.


def general_pics(dataframe,head=5):
    print("##################### head ###############")
    print(dataframe.head())
    print("######################### describe #############")
    print(dataframe.describe().T)
    print("######################### NA #################")
    print(dataframe.isnull().sum())
    print("##################### info ######################")
    print(dataframe.info())
general_pics(df,head=5)


def grab_col_names(dataframe,cat_th=10,car_th=20):
    #cat_cols, cat_but_car
    cat_cols=[i for i in dataframe.columns if dataframe[i].dtypes=="O"]
    num_but_cat=[i for i in dataframe.columns if dataframe[i].nunique() < cat_th and dataframe[i].dtypes != "O"]

    cat_but_car=[i for i in dataframe.columns if dataframe[i].nunique()>car_th and dataframe[i].dtypes == "0"]
    cat_cols=cat_cols + num_but_cat
    cat_cols=[i for i in cat_cols if i not in cat_but_car]

    #num_cols
    num_cols=[i for i in dataframe.columns if df[i].dtypes != "O"]
    num_cols=[i for i in num_cols if i not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car=grab_col_names(df)


def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),"Ratio":100*dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################")
    if plot:
        sns.countplot(x=dataframe[col_name],date=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for i in num_cols:
    num_summary(df,i,plot=True)

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Churn",col)



df["Churn"].value_counts()
df["Churn"]=df["Churn"].map({"No":0,"Yes":1})
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)



def outlier_thresholds(dataframe,col_name,q1=0.05,q3=0.95):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    interquantle_range=quartile3 - quartile1
    up_limit=quartile3 + 1.5 * interquantle_range
    low_limit=quartile1 - 1.5 * interquantle_range
    return up_limit, low_limit
outlier_thresholds(df,"tenure")

def check_outlier(dataframe,col_name):
    up_limit, low_limit=outlier_thresholds(df,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df,"tenure")

def replace_with_thresholds(dataframe,variable,q1=0.05,q3=0.95):
    up_limit, low_limit=outlier_thresholds(df,variable,q1=0.05,q3=0.95)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]=up_limit



for i in num_cols:
    print(i,check_outlier(df,i))
    if check_outlier(df,i):
        replace_with_thresholds(df,i)



df[num_cols].corr()

f,ax=plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation_Matrix",fontsize=20)
plt.show(block=True)

df.corrwith(df["Churn"]).sort_values(ascending=False)

#Feature Engineering
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df[df["TotalCharges"].isnull()]["tenure"]
df["TotalCharges"].fillna(0,inplace=True)
df.isnull().sum()


df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"New_Tenure_Year"]="0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"New_Tenure_Year"]="1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"New_Tenure_Year"]="2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"New_Tenure_Year"]="3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"New_Tenure_Year"]="4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"New_Tenure_Year"]="5-6 Year"

df["New_Engaged"]=df["Contract"].apply(lambda x: 1 if x in ["One year","Two Year"] else 0)

df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)


df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["New_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


df["New_TotalSevices"]=(df[["PhoneService", "InternetService", "OnlineSecurity",
                                       "OnlineBackup", "DeviceProtection", "TechSupport",
                                       "StreamingTV", "StreamingMovies"]]== "Yes").sum(axis=1)

df["New_Flag_Any_Streaming"]=df.apply(lambda x: 1 if (x["StreamingTV"]=="Yes") or (x["StreamingMovies"]=="Yes") else 0,axis=1)

df["New_Flag_AutoPayment"]=df["PaymentMethod"].apply(lambda x:1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["New_TotalSevices"] + 1)
df.head()


cat_cols, num_cols, cat_but_car=grab_col_names(df)

# LABEL ENCODING
le = LabelEncoder()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)


cat_cols=[i for i in df.columns if i not in binary_cols and i not in ["Churn","New_TotalSevices"]]
cat_cols

def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,cat_cols,drop_first=True)
df.head()


y=df["Churn"]
X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=17)
catboost_model=CatBoostClassifier(verbose=False,random_state=12345).fit(X_train,y_train)
y_pred=catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

#Accuracy: 0.79
#Recall: 0.67
#Precision: 0.48
#F1: 0.56
#Auc: 0.75

# Base Model
# # Accuracy: 0.7837
# # Recall: 0.6333
# # Precision: 0.4843
# # F1: 0.5489
# # Auc: 0.7282
