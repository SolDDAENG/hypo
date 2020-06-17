# Logistic Regression : 지도학습 중 분류모델(이항분류)
# 독립변수(연속형), 종속변수(범주형)
# logit변환(odds ratio의 결과에 log를 씌워 0 ~ 1 사이의 확률값을 반환)

# sigmoid function    1 / (1 + e ** -x)
# sigmoid function**매우 중요**    e : 자연상수
import math
import numpy as np


def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

    
print(sigmoidFunc(1))
print(sigmoidFunc(3))
print(sigmoidFunc(-2))
print(sigmoidFunc(0.2))
print(sigmoidFunc(0.8))
print(np.around(sigmoidFunc(-0.03)))
print(np.around(sigmoidFunc(0.8)))
print(np.rint(sigmoidFunc(0.8)))

print('car ' * 20)
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 자동차 데이터로 분류 연습 (연비와 마력수로)
mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(2))
mtcar = mtcars.loc[:, ['mpg', 'hp', 'am']]  # am : 변속기 종류(수동, 자동)
print(mtcar.head(2))
print(mtcar['am'].unique())  # [1 0]

# 연습 1 : logit()
formula = 'am ~ hp + mpg'
result = smf.logit(formula=formula, data=mtcar).fit()
print(result)
print(result.summary())
pred = result.predict(mtcar[:5])
print('예측값 : ', np.around(pred))
print('실제값 : ', mtcar['am'][:5])

# confusion matrix(혼동 행렬)
conf_tab = result.pred_table()
print(conf_tab)
print('분류 정확도(accuracy) : ', (16 + 10) / len(mtcar))  # 분류 정확도(accuracy) :  0.8125
print('분류 정확도(accuracy) : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 분류 정확도(accuracy) :  0.8125 # len(mtcar) : mtcar의 전체 개수
from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류 정확도(accuracy) : ', accuracy_score(mtcar['am'], np.around(pred2)))  # 분류 정확도(accuracy) :  0.8125
# 0.8125 ==> 81%의 정확도

# 연습 2 : glm()
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
# family = sm.families.Binomial() : 이항분포에 이항분류를 하라는 것.
print(result2)
print(result2.summary())
glm_pred = result2.predict(mtcar[:5])
print('glm_pred : ', glm_pred)
print('glm_pred : ', np.around(glm_pred))
glm_pred2 = result2.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2)))  # 분류 정확도 :  0.8125

# 머신러닝의 포용성 : 머신러닝은 수학이 아니라 추론이다.......

print()
# 새로운 값으로 예측
# newdf = mtcar.iloc[:2]
newdf = mtcar.iloc[:2].copy()  # 기존 데이터를 일부 추출해 새 객체 생성 후 분류 작업    # 원본은 바뀌지 않는다.
print(newdf)
newdf['mpg'] = [10, 30]
newdf['hp'] = [100, 120]
print(newdf)
new_pred = result2.predict(newdf)
print('새로운 데이터에 대한 분류 결과 : ', new_pred)
print('새로운 데이터에 대한 분류 결과 : ', np.around(new_pred))
print('새로운 데이터에 대한 분류 결과 : ', np.rint(new_pred))  # around나 rint나 같다.

print('-----------------')
import pandas as pd
newdf2 = pd.DataFrame({'mpg':[10, 35], 'hp':[100, 125]})
print(newdf2)
new_pred2 = result2.predict(newdf2)
print('새로운 데이터에 대한 분류 결과 : ', np.around(new_pred2))