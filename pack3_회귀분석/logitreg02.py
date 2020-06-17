# 날씨 정보 자료를 이용해 날씨 예보(내일 비 유뮤)

import pandas as pd
from sklearn.model_selection._split import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/weather.csv")
print(data.head(2), data.shape)  # (366, 12)

data2 = pd.DataFrame
data2 = data.drop(['Date', 'RainToday'], axis=1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
print(data2.head(2), data2.shape)  # (366, 10)

# RainTomorrow : 종속변수, 나머지열 : 독립변수

# train / test dataset으로 분리 : 과적함(overfitting) 방지 목적
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(data2.shape, train.shape, test.shape)  # (366, 10) (256, 10) (110, 10)

# 분류모델
my_formula = 'RainTomorrow ~ MinTemp + MaxTemp + Rainfall ...'
col_select = "+".join(train.columns.difference(['RainTomorrow']))
my_formula = 'RainTomorrow ~ ' + col_select
print(my_formula)

# 분류 모델을 위한 학습모델 생성.
# model = smf.glm(formula=my_formula, data=train, family=sm.families.Binomial()).fit()
model = smf.logit(formula=my_formula, data=train).fit()
# print(model.summary())
# print(model.params)

print('예측값 : ', np.rint(model.predict(test)[:5]))
print('실제값 : ', test['RainTomorrow'][:5])

# 분류 정확도
conf_mat = model.pred_table()  # 참고 : GLMResults' object has no attribute 'pred_table' => GRMRusults를 pred_table에 사용할 수 없다.
print(conf_mat)
print((conf_mat[0][0] + conf_mat[1][1]) / len(train))

from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('분류 정확도 : ', accuracy_score(test['RainTomorrow'], np.around(pred)))  # 분류 정확도 :  0.8727272727272727 
# accuracy_score는 실제값, 예측값 순서로 써야함. 순서가 바뀌면 안됨.
