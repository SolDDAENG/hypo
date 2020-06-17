# 다항회귀
# : 선형 가정이 어긋날 경우(정규성을 만족하지 않을 경우) 다항식 항을 추가한 후 다항회귀모델을 작성 후 사용

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 4, 7])
# plt.scatter(x, y)
# plt.show()

# 선형회귀 모델
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]  # 차원확대
# print(x)    # 2 차원으로 확대됨.

model = LinearRegression().fit(x, y)
yfit = model.predict(x)
print(yfit)  # 단순선형회귀 모델에 의한 예측값 [2.  2.8 3.6 4.4 5.2]

from sklearn.metrics import mean_squared_error
import numpy as np
lin_mse = mean_squared_error(y, yfit)
lin_rmse = np.sqrt(lin_mse)
print('평균 제곱 오차 : ', lin_mse)
print('평균 제곱근 편차 : ', lin_rmse)

# plt.scatter(x, y)
# plt.plot(x, yfit, c='red')
# plt.show()

# 모델에 유연성을 주기 위해 다항식 특징을 추가
from sklearn.preprocessing import PolynomialFeatures  # 다항회귀를 위한 모듈
poly = PolynomialFeatures(degree=3, include_bias=False)  # degree : 열의 개수, include_bias 편향여부(True, False)
xx = poly.fit_transform(x)
print(xx)
# [[  1.   1.   1.] [  2.   4.   8.] ...
# x를 나타내는 열과 x2(제곱)을 나타내는 두 번째 열과 x3(세제곱)을 나타내는 세 번째 열로 구성이 됨. 이를 통해 선형회귀를 수행한다.

model2 = LinearRegression().fit(xx, y)
yfit2 = model2.predict(xx)
print(yfit)
plt.scatter(x, y)
plt.plot(x, yfit2, c='red')
plt.show()