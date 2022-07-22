import pandas as pd
import numpy as np
import streamlit as st
import time

# ML modules
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from catboost import CatBoostClassifier, Pool, cv

st.title('Hello')

@st.cache
def load_dataset():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test



df_train, df_test = load_dataset()

df_train

### prepare data

# train = train.dropna(subset=['Embarked'])
# train = train.drop("Cabin", axis=1)
# mean = train["Age"].mean()
# train["Age"] = train["Age"].fillna(mean)
# train = train.drop(["PassengerId", "Name", "Ticket"], axis=1)
# le = LabelEncoder()
# for col in ['Sex', "Embarked"]:
#     le.fit(train[col])
#     train[col] = le.transform(train[col])

df_train.Transported = df_train.Transported.astype('int')
#df_train[['pass_grp','pass_no']]= df_train['PassengerId'].str.split('_', n = -1, expand = True)
#df_train.drop('PassengerId', axis = 1, inplace = True)

#df_test[['pass_grp','pass_no']]= df_test['PassengerId'].str.split('_', n = -1, expand = True)
#df_test.drop('PassengerId', axis = 1, inplace = True)


df_train = pd.get_dummies(df_train, columns=['HomePlanet', 'Destination'])
df_test = pd.get_dummies(df_test, columns=['HomePlanet', 'Destination'])
df_test.drop('PassengerId', axis = 1, inplace = True)
df_train.drop('PassengerId', axis = 1, inplace = True)

# ready dataset

X_train = df_train.drop(['Transported'], axis=1) # Putting feature variables into X
y_train = df_train['Transported']

def main_fit_func(algo, X_train, y_train, cv):
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, acc, acc_cv


    
st.write('Data download')
st.write(df_train.head())

def ml_lr():
    return LogisticRegression(solver="lbfgs", max_iter=500)

def ml_knc():
    return KNeighborsClassifier()

def ml_gnb():
    return GaussianNB()

def ml_SVC():
    return LinearSVC()

def ml_sgd():
    return SGDClassifier()

def ml_dtc():
    return tree.DecisionTreeClassifier()

def ml_gbt():
    return GradientBoostingClassifier()

ml_methods = {'Logistic Regression' : ml_lr(), 
              'K-Nearest Neighbours' : ml_knc(),
              'Gaussian Naive Bayes' : ml_gnb(),
              'Linear Support Vector Machines (SVC)' : ml_SVC(),
              'Stochastic Gradient Descent': ml_sgd(),
              'Decision Tree Classifier' : ml_dtc(),
              'Gradient Boost Trees' : ml_gbt()}

select_ml = st.sidebar.selectbox('Choise methods ML', list(ml_methods))
button_apply = st.sidebar.button('Apply')
if button_apply:
    start_time = time.time()
    train_pred_r, acc_r, acc_cv_r = main_fit_func(ml_methods[select_ml], 
            X_train, 
            y_train, 
            10)
    
    time_r = (time.time() - start_time)

    st.subheader(f'Result {select_ml}')
    st.write(f'Accuracy: {acc_r}')
    st.write(f'Accuracy CV 10-Fold: {acc_cv_r}')
    st.write(f'Running time: {time_r}')