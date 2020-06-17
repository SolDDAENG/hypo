# 방법 4 : stats.linregress() 를 이용 - 모델 o (모델을 만듬)

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_iq = pd.read_csv('../../../Downloads/py_hypo 6/testdata/score_iq.csv')
print(score_iq.info())
print(score_iq.head())

x = score_iq.iq     # 독립변수
y = score_iq.score  # 종속변수

# 상관계수 - 시각화
# print(np.corrcoef(x, y))    # numpy의 상관계수   0.88222
# print(score_iq.corr())      # pandas의 상관계수.   양의 상관관계로 나온다.
# plt.scatter(x, y)
# plt.show()

# 두 변수 간 인과관계가 있어 보이므로 회귀분석을 수행
# LinearRegression, ols()
model = stats.linregress(x, y)
print(model)
# LinregressResult(slope=0.6514309527270075, intercept=-2.8564471221974657, rvalue=0.8822203446134699, pvalue=2.8476895206683644e-50, stderr=0.028577934409305443)
# newx = 143.5  # iq가 143.5일 때
# yhat = 0.65143 * newx + -2.856447
# print('yhat : ', yhat)  # yhat :  90.623758
print(model.slope)      # 기울기 : 0.6514309527270075
print(model.intercept)  # 절편 : -2.8564471221974657
print(model.pvalue)     # pvalue 가 0.5 보다 작으면 독립변수로 의미 있다. 하나만 나왔기 때문.



















