# * 서로 대응인 두 집단의 평균 차이 검정(paired samples t-test)
# 처리 이전과 처리 이후를 각각의 모집단으로 판단하여, 동일한 관찰 대상으로부터 처리 이전과 처리 이후를 1:1로 대응시킨 두 집단으로 부터
# 의 표본을 대응표본(paired sample)이라고 한다.
# 대응인 두 집단의 평균 비교는 동일한 관찰 대상으로부터 처리 이전의 관찰과 이후의 관찰을 비교하여 영향을 미친 정도를 밝히는데 주로 사용
# 하고 있다. 집단 간 비교가 아니므로 등분산 검정을 할 필요가 없다.

import numpy as np
import scipy as sp
import scipy.stats as stats

# 특강이 시험 점수에 영향을 주었는지 검정하기
# 귀무 : 특강 전후의 시험 점수는 차이가 없다.
# 대립 : 특강 전후의 시험 점수는 차이가 있다.
np.random.seed(0)
x1 = np.random.normal(80, 10, 100)  # normal(평균, 표준편차, 데이터 수) : 정규분포를 따음
x2 = np.random.normal(77, 10, 100)
# print(x1)
# print(x2)

import seaborn as sns  # seaborn : matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
import matplotlib.pyplot as plt

sns.distplot(x1, kde=False, fit=stats.norm)
sns.distplot(x2, kde=False, fit=stats.norm)
plt.show()
print(stats.ttest_rel(x1, x2))
# 해석 : pvalue=0.0450057502344844 < 0.05 이므로 귀무가설 기각. 특강 전후의 시험 점수는 차이가 있다.
# 특강이 시험점수에 영향을 주었다. 라는 의견을 받아들임

#=============================================================================

# 실습) 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게 변화
baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

# 귀무 : 복부 수술 전 몸무게와 복부 수술 후 몸무게 변화가 없다.
# 대립 : 복부 수술 전 몸무게와 복부 수술 후 몸무게 변화가 있다.
p_sample = stats.ttest_rel(baseline, follow_up)
print(p_sample)
# 해석 : pvalue=0.006326650855933662 < 0.05 이므로 귀무 기각
