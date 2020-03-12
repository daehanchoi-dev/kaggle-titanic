import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas import Series

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


###   Read Data   ###

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#combine = [df_train, df_test]

###   EDA Processing   ###
"""
print(df_train.columns.values)
print(df_train.describe(include='all'))
print(df_train[['Survived','Pclass']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Survived','Sex']].groupby(['Sex'], as_index=True).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Survived','Embarked']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False))
print(pd.crosstab(df_train.Fare, df_train.Survived))
"""


## extract NAN Value on Train Data ##
"""
for col in df_train.columns:
    data = 'column:{:>10} Percent Of NAN value {:.2f}%\t' \
           .format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(data)

print(sep='\n')

## extract NAN Value on Test Data ##

for col in df_test.columns:
    data = 'column:{:>10} Percent Of NAN value {:.2f}%\t' \
           .format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(data)
"""

df_train.drop(['Ticket','Cabin'], axis=1, inplace=True)
df_test.drop(['Ticket','Cabin'], axis=1, inplace=True)


df_train['initial'] = df_train['Name'].str.extract('([A-Za-z]+)\.')
df_test['initial'] = df_test['Name'].str.extract('([A-Za-z]+)\.')
#print(df_test['initial'].value_counts())
#print(pd.crosstab(df_train['initial'], df_train['Sex']))


df_train['initial'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Sir', 'Rev'],
                            'Someone',inplace=True)

df_train['initial'].replace(['Mile', 'Ms', 'Mme','Mlle'], ['Miss', 'Miss', 'Mrs','Miss'], inplace=True)

df_test['initial'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Sir', 'Rev', 'Dona'],
                            'Someone',inplace=True)

df_test['initial'].replace(['Mile', 'Ms', 'Mme','Mlle'], ['Miss', 'Miss', 'Mrs','Miss'], inplace=True)


#print(df_train.initial.value_counts())
#print(df_test.initial.value_counts())

df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] =='Mr'), 'Age'] = 33
df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] =='Mrs'), 'Age'] = 37
df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] =='Master'), 'Age'] = 5
df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] =='Miss'), 'Age'] = 22
df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] =='Someone'), 'Age'] = 45

df_test.loc[(df_test['Age'].isnull()) & (df_test['initial'] =='Mr'), 'Age'] = 33
df_test.loc[(df_test['Age'].isnull()) & (df_test['initial'] =='Mrs'), 'Age'] = 37
df_test.loc[(df_test['Age'].isnull()) & (df_test['initial'] =='Master'), 'Age'] = 5
df_test.loc[(df_test['Age'].isnull()) & (df_test['initial'] =='Miss'), 'Age'] = 22
df_test.loc[(df_test['Age'].isnull()) & (df_test['initial'] =='Someone'), 'Age'] = 45



df_train['initial'] = df_train['initial'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Someone': 4})
df_test['initial'] = df_test['initial'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Someone': 4})

df_train= pd.get_dummies(df_train, columns=['initial'], prefix='initial')
df_test= pd.get_dummies(df_test, columns=['initial'], prefix='initial')

df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1})
df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1})


#print(df_train.iloc[1,:])

df_train.drop(['Name','PassengerId'], axis=1, inplace=True)
df_test.drop(['Name','PassengerId'], axis=1, inplace=True)

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

# print(df_train.FamilySize.value_counts())
df_train['Alone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'Alone'] = 1

#print(df_train[['Alone', 'Survived']].groupby(['Alone'],as_index=True).mean().sort_values(by='Survived', ascending=True))

df_test['Alone'] = 0
df_test.loc[df_test['FamilySize'] == 1, 'Alone'] = 1

df_train.drop(['SibSp','Parch','FamilySize'], axis=1, inplace=True)
#print(df_train.iloc[1,:])

df_test.drop(['SibSp','Parch','FamilySize'], axis=1, inplace=True)
#print(df_test.iloc[1,:])

#print(df_train.Age.isnull().sum())
#print(df_test.Age.isnull().sum())


df_train['AgeGroup'] = pd.cut(df_train['Age'], 10)
#print(df_train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=True).mean().sort_values(by='AgeGroup', ascending=True))
df_test['AgeGroup'] = pd.cut(df_train['Age'], 10)

df_train.loc[df_train['Age'] <=8, 'Age'] = 0
df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 24), 'Age' ] = 1
df_train.loc[(df_train['Age'] > 24) & (df_train['Age'] <= 32), 'Age'] = 2
df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 40), 'Age'] = 3
df_train.loc[(df_train['Age'] > 40) & (df_train['Age'] <= 48), 'Age'] = 4
df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 56), 'Age'] = 5
df_train.loc[(df_train['Age'] > 56) & (df_train['Age'] <= 64), 'Age'] = 6
df_train.loc[(df_train['Age'] > 64) & (df_train['Age'] <= 72), 'Age'] = 7
df_train.loc[df_train['Age'] > 72, 'Age'] = 8

df_test.loc[df_test['Age'] <=8, 'Age'] = 0
df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 24), 'Age' ] = 1
df_test.loc[(df_test['Age'] > 24) & (df_test['Age'] <= 32), 'Age'] = 2
df_test.loc[(df_test['Age'] > 32) & (df_test['Age'] <= 40), 'Age'] = 3
df_test.loc[(df_test['Age'] > 40) & (df_test['Age'] <= 48), 'Age'] = 4
df_test.loc[(df_test['Age'] > 48) & (df_test['Age'] <= 56), 'Age'] = 5
df_test.loc[(df_test['Age'] > 56) & (df_test['Age'] <= 64), 'Age'] = 6
df_test.loc[(df_test['Age'] > 64) & (df_test['Age'] <= 72), 'Age'] = 7
df_test.loc[df_test['Age'] > 72, 'Age'] = 8

df_train.drop(['AgeGroup'], axis=1, inplace=True)
df_test.drop(['AgeGroup'], axis=1, inplace=True)
#print(df_train.iloc[1,:])
#print(df_test.iloc[1,:])


df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
#print(df_test.Fare.isnull().sum())

df_train['FareGroup'] = pd.qcut(df_train['Fare'], 5)
df_test['FareGroup'] = pd.qcut(df_test['Fare'], 5)
#print(df_train[['FareGroup', 'Survived']].groupby(['FareGroup'],as_index=True).mean().sort_values(by='FareGroup', ascending=True ))

df_train.loc[df_train['Fare'] <= 7.854, 'Fare'] = 0
df_train.loc[(df_train['Fare'] > 7.854) & (df_train['Fare'] <= 10.5), 'Fare' ] = 1
df_train.loc[(df_train['Fare'] > 10.5) & (df_train['Fare'] <= 21.679), 'Fare' ] = 2
df_train.loc[(df_train['Fare'] > 21.679) & (df_train['Fare'] <= 39.688), 'Fare' ] = 3
df_train.loc[(df_train['Fare'] > 39.688) & (df_train['Fare'] <= 512.329), 'Fare' ] = 4

df_test.loc[df_test['Fare'] <= 7.854, 'Fare'] = 0
df_test.loc[(df_test['Fare'] > 7.854) & (df_test['Fare'] <= 10.5), 'Fare' ] = 1
df_test.loc[(df_test['Fare'] > 10.5) & (df_test['Fare'] <= 21.679), 'Fare' ] = 2
df_test.loc[(df_test['Fare'] > 21.679) & (df_test['Fare'] <= 39.688), 'Fare' ] = 3
df_test.loc[(df_test['Fare'] > 39.688) & (df_test['Fare'] <= 512.329), 'Fare' ] = 4

df_train.drop(['FareGroup'], axis=1, inplace=True)
df_test.drop(['FareGroup'], axis=1, inplace=True)

############################################  EDA Processing Complete  ######################################

############################################  Apply scikit-learn Model ######################################

x_train = df_train.drop(['Survived'], axis=1).values
y_train = df_train['Survived'].values
x_test = df_test.values

#print(x_train.shape, y_train.shape, x_test.shape)

x_tr, x_vld, y_tr, y_vld = train_test_split(x_train, y_train, test_size=0.3, random_state=2018)

#  Logistic Regression

loreg = LogisticRegression()
loreg.fit(x_tr, y_tr)
lo_predict = loreg.predict(x_test)
acc_loreg = round(loreg.score(x_tr, y_tr) * 100, 2)
#print("Logistic Regression 확률{}".format(acc_loreg))

#  RandomForest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_tr, y_tr)
forest_predict = random_forest.predict(x_test)
random_forest.score(x_tr, y_tr)
acc_random_forest = round(random_forest.score(x_tr, y_tr) * 100, 2)
#print("RandomForest 확률{}".format(acc_random_forest))


#  Support Vector Machine

svc = SVC()
svc.fit(x_tr, y_tr)
svm_predict = svc.predict(x_test)
acc_svc = round(svc.score(x_tr, y_tr) * 100, 2)
#print("SVM 확률{}".format(acc_svc))

#  K-Nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_tr, y_tr)
knn_predict = knn.predict(x_test)
acc_knn = round(knn.score(x_tr, y_tr) * 100, 2)
#print("KNN 확률{}".format(acc_knn))


#  Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_tr, y_tr)
gau_predict = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_tr, y_tr) * 100, 2)
#print("Gaussian 확률{}".format(acc_gaussian))

#  Decision_Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_tr, y_tr)
dec_predict = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_tr, y_tr) * 100, 2)
#print("Decision Tree 확률{}".format(acc_decision_tree))

#  Sort Each accuracy for models
models = pd.DataFrame({
    'Model': ['Logistic Regression','Random Forest','Support Vector Machines',
              'K-Nearest Neighbor', 'Gaussian Naive Bayes', 'Decision Tree'],
    'Score': [acc_loreg, acc_random_forest, acc_svc,
              acc_knn, acc_gaussian, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

# Coefficient Logistic Regression

coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(loreg.coef_[0])

#print("Logistic Regression Coefficient\n", coeff_df.sort_values(by='Correlation', ascending=True))

# Coefficient Random Forest

feature_importance = random_forest.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)

#print("RandomForest Coefficient\n" , Series_feat_imp)