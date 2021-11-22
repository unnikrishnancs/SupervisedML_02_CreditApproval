#! /usr/bin/python3

#import packages
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,LabelBinarizer,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score
from matplotlib import pyplot as plt
from sklearn.neighbors  import KNeighborsClassifier


#import dataset...replace "?" with nan
data=pd.read_csv("credit_approval.data",na_values=["?"])

data.columns =["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

#print first 5 rows
print(data.head(150))

# ================================
#perform exploratory data analysis
# ================================
data.info()

#distribution of values in categorical columns
for i in data.select_dtypes(["object"]):
 print(data.value_counts([i]))

#change data types if necc
#data["A2"].astype(float)

# ==================
#data pre-processing (handle missing values, categorical to int etc)
# ==================

#validate imported column data types, non-null count etc
data.info()

#handle missing values
data["A1"].fillna(data["A1"].mode()[0],inplace=True) #replace with mode()
data["A2"].fillna(data["A2"].median(),inplace=True) #replace with median() 
data["A14"].fillna(data["A14"].median(),inplace=True) #replace with median(

#drop null values in col. A4,A5,A6 and A7
data.dropna(inplace=True)

#see if changes applied
data.info()

#convert categorical to int for X
ohe=OneHotEncoder()

ohe.fit(data[["A1","A4","A5","A6","A7","A9","A10","A12","A13"]])
print(ohe.categories_)
print(ohe.get_feature_names())
data_trans=ohe.transform(data[["A1","A4","A5","A6","A7","A9","A10","A12","A13"]]).toarray()
data_trans_df=pd.DataFrame(data=data_trans,columns=ohe.get_feature_names())
print(data_trans_df)

#convert categorical to int for y
lb=LabelBinarizer()
lb.fit(data["A16"])
print(lb.classes_)
y_trans=lb.transform(data["A16"])
y_trans=pd.Series(y_trans.ravel())

#merge dataframes into one
df_final=pd.concat([data_trans_df,data.iloc[:,[1,2,7,10,13,14]],y_trans],axis=1)

#after mergin, drop rows with any null values
df_final.dropna(inplace=True)
print(df_final)

# ==============
# Define X and y
# ==============

#input features
X=df_final.iloc[:,:-1]
print(X.iloc[:5,:])

#dependent variable
y=df_final.iloc[:,-1]
print(y[:5])

#apply feature scaling
scl=StandardScaler()
scl.fit(X)
print(scl.mean_)
X_scale=scl.transform(X)
print(X_scale)

#split into train and test set
x_train,x_test,y_train,y_test=train_test_split(X_scale,y,test_size=0.2,random_state=10)

# ==========
#train model
# ==========
model1=LogisticRegression()
model2=KNeighborsClassifier()


model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

# ===============
#predict test data
# ===============
y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)

#predict probabilities
print(model1.classes_)
lr_probs=model1.predict_proba(x_test)
knn_probs=model2.predict_proba(x_test)


# ==================
#performance metrics
# ==================
print("\n")
print("Confusion Matrix (Logistic Regression)")
print(confusion_matrix(y_pred1,y_test))

print("\n")
print("Accuracy Score (Logistic Regression)")
print(accuracy_score(y_pred1,y_test))

print("\n")
print("Confusion Matrix (KNN)")
print(confusion_matrix(y_pred2,y_test))

print("\n")
print("Accuracy Score (KNN)")
print(accuracy_score(y_pred2,y_test))

# ==================
#plot roc curve
# ==================

#generate a "No skill" prediction (majority class))
ns_probs1=[0 for _ in range(len(y_test))]
#ns_probs2=[0 for _ in range(len(y_test))]

#keep probabilities for the "positive outcome" only (i:e class 1)
r_probs=lr_probs[:,1]
k_probs=knn_probs[:,1]

#calc area under curve (auc) score...'no skill'
ns_auc1=roc_auc_score(y_test,ns_probs1)
#ns_auc2=roc_auc_score(y_test,ns_probs2)

#calc area under curve (auc) score...'logistic regression'
lr_auc=roc_auc_score(y_test,r_probs)
knn_auc=roc_auc_score(y_test,k_probs)

#summarize scores
print("\n")
print("ROC AUC scores (Logistic Regression)")
print("No Skill: ROC AUC=%.3f"%(ns_auc1))
print("Logistic: ROC AUC=%.3f"%(lr_auc))
print("\n")
print("ROC AUC scores (KNN)")
print("No Skill: ROC AUC=%.3f"%(ns_auc1))
print("Logistic: ROC AUC=%.3f"%(knn_auc))

#calc roc curves..returns FPR and TPR
# Logistic Regression
ns_fpr1,ns_tpr1,ns_thresholds1=roc_curve(y_true=y_test,y_score=ns_probs1)
print(f"'no skill' thresholds \n {ns_thresholds1}")
lr_fpr,lr_tpr,lr_thresholds=roc_curve(y_true=y_test,y_score=r_probs)
print(f"'logistic regression' thresholds \n {lr_thresholds}")

# KNN
#ns_fpr2,ns_tpr2,ns_thresholds2=roc_curve(y_true=y_test,y_score=ns_probs2)
#print(f"'no skill' thresholds \n {ns_thresholds2}")
knn_fpr,knn_tpr,knn_thresholds=roc_curve(y_true=y_test,y_score=k_probs)
print(f"'KNN' thresholds \n {knn_thresholds}")



#plot roc curve for model
plt.plot(ns_fpr1,ns_tpr1,linestyle="--",label="No Skill")
plt.plot(lr_fpr,lr_tpr,marker=".",label="Logistic")
#plt.plot(ns_fpr,ns_tpr,linestyle="--",label="No Skill")
plt.plot(knn_fpr,knn_tpr,marker="+",label="KNN")

#axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

#show legend
plt.legend()

#show
plt.show()
