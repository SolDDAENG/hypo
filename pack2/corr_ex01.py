# 상관분석 (r) : 두 개 이상의 변수 간에 어떤 관계가 있는지 분석하는 것
# 공분산을 표준화함 -1 ~ 0 ~ 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'id1': (1, 2, 3, 4, 5), 'id2': (2, 3, -1, 7, 9)})
print(df)
# plt.scatter(df.id1, df.id2)
# plt.show()

print('공분산 : ', df.cov())  # 관련 방향성은 제시하나 강도 표현은 모호하다.
print('상관계수 : ', df.corr())  # 공분산의 약점을 해결

print('\n상품의 만족도 관계 확인 ----------------------------')
data = pd.read_csv('../testdata/drinking_water.csv')
print(data.head())
print(data.describe())

# 각 요소의 표준편차 출력
print(np.std(data.친밀도))     # 0.968505126935272
print(np.std(data.적절성))     # 0.8580277077642035
print(np.std(data.만족도))     # 0.8271724742228969

# 시각화
# plt.hist([np.std(data.친밀도), np.std(data.적절성), np.std(data.만족도)])
# plt.show()

# 공분산 출력
print(np.cov(data.친밀도, data.적절성))   # numpy는 np.cov(변수1, 변수2)
print(np.cov(data.친밀도, data.만족도))
print(data.cov())   # DataFrame은 여러 개를 한번에 입력 가능하다.
print()

# 상관계수 출력
print(np.corrcoef(data.친밀도, data.적절성))  # numpy는 np.corrcoef(변수1, 변수2)
print(np.corrcoef(data.친밀도, data.만족도))

print(data.corr())
print(data.corr(method='pearson'))  # 변수가 등간 / 비율 척도 일 때. 정규성을 따름. 기본값이 pearson
print(data.corr(method='spearman')) # 변수가 서열 척도 일 때. 정규성을 따르지 않음
print(data.corr(method='kendall'))  # spearman과 유사

# 상관관계를 시각화(heatmap 색으로 표현)
import seaborn as sns
plt.rc('font', family='DejaVu Sans')
sns.heatmap(data.corr())
plt.show()