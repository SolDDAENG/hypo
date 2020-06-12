# 두 집단의 가설검정 – 실습 시 분산을 알지 못하는 것으로 한정하겠다.
# * 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.

import pandas as pd
import scipy.stats as stats
from numpy import average

# 실습) 남녀 두 집단 간 파이썬 시험의 평규에 차이가 있는지 검정
# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.
# 대립 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 있다.

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

# two_samples = stats.ttest_ind(male, female)
two_sample = stats.ttest_ind(male, female, equal_var=True)  # 등분산성 만족(두 그룹의 분산이 같다) default : equal_var=True
print(two_sample)
# Ttest_indResult(statistic=1.233193127514512, pvalue=0.2525076844853278)
sta, pv = two_sample
print('sta : ', sta)
print('pv : ', pv)
print(average(male), ' ', average(female))
# 해석 : male 평균 83.8, female 평균 72.24
# p-value=0.25250 > 0.05 이므로 귀무가설 채택.    남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.

# 실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv
# 귀무 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# 대립 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.
data = pd.read_csv('../testdata/two_sample.csv')
print(data.head(3))
result = data[['method', 'score']]
print(result[:3])
print()

# 데이터 분리
m1 = result[result['method'] == 1]  # 방법 1
m2 = result[result['method'] == 2]  # 방법 2

score1 = m1['score']
score2 = m2['score']
# print(score1)
# print()
# print(score2)

# NaN은 평균으로 대체
# NaN 확인 방법 : describe(), isnull().sum(), isnull().any() ...

sco1 = score1.fillna(score1.mean())
sco2 = score2.fillna(score2.mean())

# 분포를 시각화하여 정규성 확인
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(sco1, kde=False, fit=stats.norm)
sns.distplot(sco2, kde=False, fit=stats.norm)
plt.show()

# 정규성 확인 함수 - shapiro : 0.05보다 크면 정규성을 띈다.
print(stats.shapiro(sco1))  # (0.965552806854248, 0.3679903745651245)    0.3679 > 0.05 정규성 분포를 이룸
print(stats.shapiro(sco2))  # (0.9621098637580872, 0.6714189648628235)    0.6714 > 0.05 정규성 분포를 이룸

# 등분산성  : 0.05보다 크면 등분산성을 이룸.
print(stats.levene(sco1, sco2))  # 가장 일반적인 등분산성 함수.   0.4568 > 0.05 이므로 등분산성을 따름
# print(stats.levene(sco1, sco2).pvalue)
print(stats.fligner(sco1, sco2))
print(stats.bartlett(sco1, sco2))

print(stats.ttest_ind(sco1, sco2))
print(stats.ttest_ind(sco1, sco2, equal_var=True))      # equal_var : 등분산성 만족한 경우. 기본값.
# Ttest_indResult(statistic=-0.19649386929539883, pvalue=0.8450532207209545)
# 해석 : p-value(0.8450) > 0.05 이므로 귀무 채택.    두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.

print(stats.ttest_ind(sco1, sco2, equal_var=False))  # equal_var : 등분산성 만족하지 못한 경우.

# 만약 정규성 만족 못할 경우
# stats.wilcoxon()    # 이걸 사용하자. 다만 두개의 길이가 같아야 실행된다.
# stats.kruskal()
# stats.mannwhitneyu()

