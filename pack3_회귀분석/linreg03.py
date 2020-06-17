# 사용 빈도가 높은 회귀분석 모델 ols() - 가장 기본적인 결정론적 선형회귀 방법
import pandas as pd

df = pd.read_csv('../../../Downloads/py_hypo 6/testdata/drinking_water.csv')
print(df.head(3))
print(df.corr())    # 피어슨 상관계수 확인

import statsmodels.formula.api as smf

# 단순 선형회귀 (독립변수가 1개)
model = smf.ols(formula='만족도 ~ 적절성',data=df).fit()    # r style의 모델
# print(model.summary())  
# R-squared(결정계수, 설명력): 0.588 (r을 제곱한 값)
# Prob (F-statistic): 2.24e-52 (F-statistic으로 유추된 모델 전체의 pvalue. 0.05보다 작으면 유효한 모델로 취급)
# [0.025      0.975] : 95%의 신뢰구간
# P>|t| : 각각 독립변수의 pvalue
print('\n회귀계수 : ', model.params)    # Intercept(절편) : 0.778858   적절성(기울기) : 0.739276
print('\n결정계수(설명력) : ', model.rsquared)
print('\np값 : ', model.pvalues) # p-value : 적절성(모델 전체에 대한 p-value) 2.235345e-52
#print('전체 예측값 : ', model.predict()) # 예측값 :  [3.73596305 2.99668687 3.73596305 ...
print('\n',df.만족도[0], model.predict()[0]) # 실제값 : 3, 예측값 : 3.7359630488589186


# 시각화 
import matplotlib.pylab as plt  # pyplot이 아닌 pylab도 가능
import numpy as np

plt.scatter(df.적절성, df.만족도) # 산점도 표현 
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)
print('slope, intercept : ', slope, intercept)  # 기울기, 절편
plt.plot(df.적절성, slope * df.적절성 + intercept, 'b')    # plot(실제값, 예측값)    y = wx + b
plt.show()


# 다중 선형회귀
print('\n다중 선형 회귀-------------------------------------------------------------------')
model2 = smf.ols(formula='만족도 ~ 적절성 + 친밀도', data=df).fit()  # r style의 모델
print(model2.summary()) # 다중 선형회귀 계산식 : yhat = x1*w1 + x2*w2 ... + b

























