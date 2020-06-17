# 일원분산분석 (one way anova) : 집단 구분 요인 1 ========================================================
import scipy.stats as stats
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import urllib.request

# 그룹별(3개) 시험점수의 차이 검정
# 귀무 : 그룹별(3개) 시험점수의 차이가 없다. H0
# 대립 : 그룹별(3개) 시험점수의 차이가 있다. H1

print('일원분산분석 ===========================================================================')
url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')    # numpy로 읽음
print(data, type(data)) # [243.   1.]] ... <class 'numpy.ndarray'>

#data2 = pd.read_csv(urllib.request.urlopen(url))    # pandas로 읽음.
#print(data2.head(3), type(data2))   # <class 'pandas.core.frame.DataFrame'>

print()
gr1 = data[data[:,1] == 1, 0]   # 집단 1만 들어가있다.
#print(gr1)
gr2 = data[data[:,1] == 2, 0]   # 집단 2만 들어가있다.
gr3 = data[data[:,1] == 3, 0]   # 집단 3만 들어가있다.
print(stats.shapiro(gr1)[1])   # 정규성 확인 -> 0.3336853086948395 > 0.05 정규성 만족
print(stats.shapiro(gr2)[1])   # 정규성 확인 -> 0.6561065912246704 > 0.05 정규성 만족
print(stats.shapiro(gr3)[1])   # 정규성 확인 -> 0.832481324672699 > 0.05 정규성 만족

# 그룹간 데이터 들의 분포를 시각화
# plot_data = [gr1, gr2, gr3]
# plt.boxplot(plot_data)
# plt.show()


# 일원분산분석 방법 1
f_statistic, p_val = stats.f_oneway(gr1, gr2, gr3)
print('일원분산분석 결과 : f_statistic=%f, p_val=%f'%(f_statistic, p_val))
# 일원분산분석 결과 : f_statistic=3.711336, p_val=0.043589 < 0.05 이므로 귀무 기각.
# 그룹별(3개) 시험점수는 차이가 있다. 라는 의견이 통계적으로 유의하다.


# 일원분산분석 방법 2 : Linear Model을 속성으로 사용
df = pd.DataFrame(data, columns=['value','group'])
print(df.head(3))
lmodel = ols('value ~ C(group)', df).fit() # 종속변수 value에 독립변수 group영향을 받고, 데이터값은 df, .fit()을 해서 학습한다. 
                                            # C(그룹칼럼...) : 범주형임을 명시적으로 표시
print(anova_lm(lmodel))    # PR(>F) <= p-vlaue 0.043589 < 0.05 이므로 귀무 기각




# 이원분산분석 (two way anova) : 집단 구분 요인 2 ========================================================
print('\n\n이원분산분석 ===========================================================================')
url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(url)
print(data.head(3))
print(data.tail(3))

# 귀무 : 관측자와 태아수 그룹에 따라 태아 머리둘레의 차이가 없다.
# 대립 : 관측자와 태아수 그룹에 따라 태아 머리둘레의 차이가 있다.

# 시각화
# plt.rc('font', family='malgun gothic')
# data.boxplot(column='머리둘레', by='태아수', grid=True)
# plt.show()    # 태아의 머리둘레는 차이가 있어 보임. 관측자와 상호 작용이 있는지 분산분석으로 검정
formula = '머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)'    # C(관측자수) + C(태아수):C(관측자수) : 상호 작용으로 검정. :부터 상호작용
lm = ols(formula = formula, data=data).fit()
print(anova_lm(lm))
# 해석 : 
# C(태아수)     PR(>F) p-value : 1.051039e-27 < 0.05 이므로 머리둘레의 차이가 있다.
# C(관측자수)    PR(>F) p-value : 6.497055e-03(대략 0.006) < 0.05 이므로 귀무 기각. 대립 채택. 머리둘레의 차이가 있다.
# C(태아수):C(관측자수) PR(>F) p-value : 3.295509e-01 > 0.05 이므로 상호작용에 의한 머리둘레의 차이가 없다.
# 결론 : 관측자수와 태아수는 머리둘레의 영향을 미치나, 관측자수와 태아수에 상효작용에 의한 영향은 없다.

print('\n상호작용 없이 확인')
formula2 = '머리둘레 ~ C(태아수) + C(관측자수)'    
lm2 = ols(formula = formula2, data=data).fit()  # 상호 작용 x
print(anova_lm(lm2))
# 





















