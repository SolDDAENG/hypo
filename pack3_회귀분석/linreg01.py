# 선형회귀 : 독립변수(연속), 종속변수(연속) ( linear regression )
# 회귀분석은 각 데이터에 대한 잔차제곱합이 최소가 되는 선형회귀식을 도출하는 방법
# 맛보기

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np

np.random.seed(12)

# 방법 1 : make_regression() 을 이용 - 모델x (모델이 만들어지지 않음)
print('\n방법 1 : make_regression()-------------------------------------------0----------')
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)  # coef : 기울기, n_features : 독립변수, n_samples : 샘플데이터, bias : 절편
print(x[:5])  # sample 독립변수 자료
print(y[:5])  # sample 종속변수 자료. 독립변수 -1.700이 들어가면 종속변수 -52.172가 나와야 한다.
print(coef)  # 기울기
# y = wx(기울기) + b(절편)    y = 89.47430739278907 * x + 100
yhat = 89.47430739278907 * -1.70073563 + 100  # 기울기 * x + 100
print('yhat : ', yhat)
yhat = 89.47430739278907 * -0.67794537 + 100  # 기울기 * x + 100
print('yhat : ', yhat)

new_x = 0.5
pred_yhat = 89.47430739278907 * new_x + 100
print('pred_yhat : ', pred_yhat)

xx = x  # n_sample로 만든 50개의 데이터
yy = y

# 방법 2 : make_LinearRegression() 을 이용 - 모델 o (모델을 만듬)
print('\n방법 2 : make_LinearRegression()------------------------------------------------')
from sklearn.linear_model import LinearRegression

model = LinearRegression()
print(model)
fit_model = model.fit(xx, yy)  # fit()으로 데이터를 학습하여 최적의 모형(잔차가 최소가되는 모형)을 추정함.
print(fit_model.coef_)  # 기울기. _는 system이 제공    [89.47430739]
print(fit_model.intercept_)  # 절편     100.0

# 예측값 구하기 1 : 수식을 직접 적용
new_x = 0.5
pred_yhat2 = fit_model.coef_ * new_x + fit_model.intercept_
print('pred_yhat2 : ', pred_yhat2)  # pred_yhat2 :  [144.7371537]

# 예측값 구하기 2 : LinearRegression이 지원하는 predict()
pred_yhat3 = fit_model.predict([[new_x]])  # 2차원 데이터로 학습했기 때문에 2차원으로 만들고 출력값이 1차원(array)이다.    여러 개의 데이터를 넣을 수도 있다.
print('pred_yhat3 : ', pred_yhat3)  # pred_yhat3 :  [144.7371537]

# 예측값 구하기 3 : predict() 예측값이 많은 데이터
x_new, _, _ = make_regression(n_samples=5, n_features=1, bias=100, coef=True)
print(x_new)  # [[-0.5314769 ] [ 0.86721724] [-1.25494726] [ 0.93016889] [-1.63412849]]
pred_yhat4 = fit_model.predict(x_new)
print('pred_yhat4 : ', pred_yhat4)  # pred_yhat4 :  [ 52.44647266 177.59366163 -12.28553693 183.2262176  -46.21251453]

# 방법 3 : ols() 를 이용 - 모델 o (모델을 만듬)
print('\n방법 3 : ols() -----------------------------------------------------------------')
import statsmodels.formula.api as smf
import pandas as pd
print(xx.shape)  # (50, 1) 2차원
x1 = xx.flatten()  # flatten() : 차원 축소. 2차원 -> 1차원
print(x1[:5], x1.shape)  # (50,) 1차원
y1 = yy

data = np.array([x1, y1])
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(3))

model2 = smf.ols(formula='y1 ~ x1', data=df).fit()  # '종속변수 ~ 독립변수' 를 fit(학습한다)
print(model2.summary())  # summary : OLS Regression Results 표 생성
# Intercept(절편) : 100.00, x1(기울기) : 89.4743

print(df[:3])  # [[-1.700736  -52.172143], [-0.677945   39.341308], [0.318665  128.512356]]
print(model2.predict()[:3])  # 기존 데이터에 대한 예측값
# [-52.17214291  39.34130801 128.51235594]

# 새로운 값에 대한 예측 결과
# print('x1 : ', x1[:2]) # -1.700736 -0.677945
newx = pd.DataFrame({'x1':[0.5, -0.8]})
predy = model2.predict(newx)
print('예측 결과 : \n', predy)  # 예측 결과 : 144.737154  28.420554

# 방법 4는 코드가 너무 길어지므로 linreg02에서 계속

