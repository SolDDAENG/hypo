# 집단 간 차이분석 : 평균 또는 비율 차이를 분석 : 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.

# * T-test와 ANOVA의 차이
# - 두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설검정을 한다.
# - 세 집단 이상의 변수에 대한 평균차이를 검정할 경우에는 ANOVA를 이용하여 검정통계랑 F값을 구해 가설검정을 한다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
# seaborn : Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지이다.

# one samples t-test
# 어느 한 집단의 평균은 0인지 검정하기(난수 사용)
# 귀무 : 자료들의 평균은 0이다.
# 대립 : 자료들의 평균은 0이 아니다.
np.random.seed(123)
mu = 0
n = 10  # 데이터가 많아질 수록 0에 가까워진다. ex) 1000, 10000 등
x = stats.norm(mu).rvs(n)  # norm : 정규분포, rvs : 랜덤 표본 생성
print(x, np.mean(x))  # mean : -0.26951611032632805

# sns.distplot(x, kde=False, rug=True, fit=stats.norm)  # 시각화  # kde=False을 넘겨주면 밀도 그래프를 그리지 않는다.
# plt.show()

result = stats.ttest_1samp(x, popmean=0)    # (데이터, 예상평균값)
print('result : ', result)
# Ttest_1sampResult(statistic=-0.6540040368674593, pvalue=0.5294637946339893) statistic : 검정평균
# 해석 : p-value(0.529463) > 0.05(유의수준) 이므로 귀무 채택.    자료들의 평균은 0이다.

# * 단일 모집단의 평균에 대한 가설검정(one samples t-test)
# 실습 예제 1)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv'
# 귀무 : 국어 점수의 평균은 80이다.
# 대립 : 국어 점수의 평균은 80이 아니다

data = pd.read_csv('../testdata/student.csv')
print(data.head())
print(data.describe())
result2 = stats.ttest_1samp(data.국어, popmean=80)
print('result2 : ', result2)
# result2 :  Ttest_1sampResult(statistic=-1.3321801667713213, pvalue=0.19856051824785262)
# 해석 : pvalue( 0.1985 ) > 0.05    # 귀무 채택. 국어 점수의 평균은 80이다.    # 실제 80이 안되지만 통계학적 수식에는 맞대요.

print('==============================')
# 실습 예제 2)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv'
# 여아 신상아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

# 귀무 : 여아 신생아의 몸무게는 평균이 2800(g)이다.
# 대립 : 여아 신생아의 몸무게는 평균이 2800(g)보다 크다.
data = pd.read_csv('../testdata/babyboom.csv')
print(data.head())
print(data.describe())
fdata = data[data.gender == 1]
print(fdata.head(), len(fdata))
print(fdata.describe())
print(np.mean(fdata.weight))    # 3132.4444444444443

# 시각화
sns.distplot(fdata.iloc[0:, 2], fit=stats.norm)
plt.show()
stats.probplot(fdata.iloc[:, 2], plot=plt)  # Q-Q plot
plt.show()

print(stats.shapiro(fdata.iloc[:, 2]))  # 정규성 확인 p(0.01798) < 0.05 이므로 정규성을 따르지 않음.
# 참고 : 정규성을 띄지 않으나 집단이 하나이므로 wilcox 검정은 할 수 없다.
result3 = stats.ttest_1samp(fdata.weight, popmean=2800)
print('result3 : ', result3)
# test_1sampResult(statistic=2.233187669387536, pvalue=0.03926844173060218)
# 해석 : pvalue(0.0392) < 0.05 이므로 귀무 기각.
# 여아 신생이의 몸무게는 평균이 2800(g)보다 크다. 라는 주장을 받아들임.
