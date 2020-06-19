# 비 예보
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/weather.csv')
print(df.head(3))
print(df.info())
x = df[['MinTemp','MaxTemp','Rainfall','Cloud']]
print(df['RainTomorrow'].unique())  # ['Yes' 'No']
#label = df['RainTomorrow'].apply(lambda x:1 if x == 'Yes' else 0)   # apply() : 함수를 실행하는 함수
label = df['RainTomorrow'].map({'Yes':1, 'No':0})
# print(label[:5])

# train / test (7:3) : overfitting 방지
x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.3, random_state=123)
print(len(x_train), ' ', len(x_test))   # 256   110


# predict
gmodel = GaussianNB()
gmodel.fit(x_train, y_train)
pred = gmodel.predict(x_test)
acc = sum(y_test == pred) / len([pred])
print('정확도 : ', acc)    # 정확도 :  93.0
print('정확도 : ', accuracy_score(y_test, pred))   # 정확도 :  0.8454545454545455
print('정확도 : ', accuracy_score(y_test, pred))   # 정확도 :  0.8454545454545455
cl_report = metrics.classification_report(y_test, pred)
print('분류 보고서 : \n', cl_report)


# k-fold(교차 검증) - 모델 학습시 입력자료를 k겹으로 나누어 학습과 검증을 함께하는 방법. 
# 데이터의 양이 작을 때 사용. 또는 train / test를 해도 문제가 있을 경우 교차검증을 위해 사용한다.
from sklearn import model_selection
# cross_val = model_selection.cross_val_score(gmodel, x, label, cv=5) # 원본 x, y를 사용
cross_val = model_selection.cross_val_score(gmodel, x_train, y_train, cv=5) # train data x, y를 사용
print(cross_val)
print(cross_val.mean()) # 0.8049773755656109

# 새로운 자료로 예측
print('\n새로운 자료로 예측 ----------------')
print(x.head(3))
import numpy as np
# MinTemp  MaxTemp  Rainfall Cloud
my_weather = np.array([[14.0, 26.9, 3.6, 3], [2.0, 11.9, 9.6, 30], [19.0, 30.9, 2.6, 2]])
print(gmodel.predict(my_weather))   # [0, 0, 0]





