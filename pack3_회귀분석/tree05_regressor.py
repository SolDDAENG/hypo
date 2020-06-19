# DecisionTreeRegressor, RandomForestRegressor ... 등의 모델로 연속형 자료 예측
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score    # 결정계수(설명력) - 예측일 때 설명력이 필요. 상관계수를 제곱
from matplotlib import style

boston = load_boston()
#print(boston.DESCR)     # CRIM,ZN,INDUS... : 독립변수    MEDV : 종속변수(집값)

# 상관관계 보기
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([df_x, df_y], axis=1)    # 행 기준 병합
print(df.head(3))
print(df.corr())

# 상관관계가 높은 항목을 선택
cols = ['MEDV', 'RM', 'PTRATIO', 'LSTAT']   # 종속변수 'MEDV'와 상관관계가 강한 열 일부(3개) 선택.
# sns.pairplot(df[cols])
# plt.show()

x = df[['LSTAT']].values   # 상관관계가 가장 강한 독립변수 LSTAT(하위층의 비율)    2차원
y = df['MEDV'].values    # 종속변수 MEDV(집값)    1차원
print(x[:3])
print(y[:3])

# 실습 1 : DecisionTreeRegressor =========================================================
print('\n실습 1 : DecisionTreeRegressor--------------------------------------------------')
model = DecisionTreeRegressor(max_depth=3).fit(x, y)
print('예측값 : ', model.predict(x)[:3])   # 예측값 :  [30.47142857 25.84701493 37.315625  ]
print('실제값 : ', y[:3].ravel())  # 실제값 :  [24.  21.6 34.7]    .ravel() : 차원을 한단계 떨어트린다.

r2 = r2_score(y, model.predict(x))
print('설명력(결정계수) : ', r2)   # 설명력(결정계수) :  0.699 약 70% 의 설명력을 가지고 있다. (이정도면 좋다)


# 실습 2 : RandomForestRegressor =========================================================
print('\n실습 2 : RandomForestRegressor--------------------------------------------------')
model2 = RandomForestRegressor(n_estimators=1000, criterion='mse').fit(x, y)     # 예측할때는 mse(평균제곱오차)를 사용한다. n_estimators : 앙상블
print('예측값 : ', model2.predict(x)[:3])   # 예측값 :  [24.6488 22.0799 35.3009]
print('실제값 : ', y[:3].ravel())  # 실제값 :  [24.  21.6 34.7]

r2r = r2_score(y, model2.predict(x))
print('설명력(결정계수) : ', r2r)   # 설명력(결정계수) :  0.9094 약 90% 의 설명력을 가지고 있다. 


# 시각화
style.use('seaborn-talk')   # 좀더 크고 선명한 차트 그리기 - 차트 서식부여
plt.scatter(x, y, c='lightgray', label='train data')
plt.scatter(x, model2.predict(x), c='r', label='predict data')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.show()


# 새 값을 예측
import numpy as np
print(x[:3])    # [[4.98] [9.14] [4.03]]
x_new = np.array([[10], [15], [1]])
print('예상 집 값 : ', model2.predict(x_new))   # 예상 집 값 :  [20.47815889 18.06333417 48.2175 ]























