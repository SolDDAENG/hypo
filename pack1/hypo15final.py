# DB 자료를 읽어 이를 근거로 가설검정 총정리 (카이제곱, t-검정, anova)
import MySQLdb
import pandas as pd
import numpy as np
import ast  # mariadb.txt를 읽어서 dict타입으로 바꿔주기 위해 사용
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols # ols 
import statsmodels.api as sm    

with open('mariadb.txt', 'r') as f:
    config = f.read()
    
config = ast.literal_eval(config)   # Dict 타입으로 변환시켜줌.
#print(config)

conn = MySQLdb.connect(**config)    # dict타입의 데이터를 줄때 **을 써야한다.
cursor = conn.cursor()  # sql문을 구사 가능한 객체 생성

sql = '''
    select jikwon_no, jikwon_name, jikwon_jik, jikwon_pay
    from jikwon
    where jikwon_jik = '과장' and jikwon_pay >= 7500
'''
cursor.execute(sql)

for data in cursor.fetchall():
    print('%s %s %s %s'%data)   # 자동으로 1대1 매핑

# 일원카이제곱 : 적합도검사, 선호도검사 등을 한다.
# 이원카이제곱 : 교차표분석, 분석대상의 집단 수에 의해서 독립성 검정과 동질성 검정 등을 한다.
print('\n\n===교차분석(이원카이제곱 검정)===================================================================')
df = pd.read_sql('select * from jikwon', conn)
print(df.head(3), df.shape)  # (30, 8)

print('각 부서(범주형)와 직원평가점수(범주형) 간의 관련성 분석(이원카이제곱) - 귀무 : 관련이 없다.')
buser = df['buser_num']
rating = df['jikwon_rating']
ctab = pd.crosstab(buser, rating)   # 교차표 작성 - 빈도수를 보여줌.
print(ctab)
chi, p, df, exp = stats.chi2_contingency(ctab)
print('chi : {:.3f}, p : {:.3f}, df : {}'.format(chi, p, df)) 
# chi : 7.339, p : 0.291, df(자유도) : 6    # :.3f => 소수 3번 째 까지만 출력 
# 해석 1 : 카이제곱표에서 임계치 12.59 > chi 7.339 이므로 귀무 채택.
# 해석 2 : p-value : 0.293 > 0.05 이므로 귀무 채택. 각 부서와 직원 평가점수간에 관련이 없다.

print('\n\n===두 집단 이하의 평균 차이 분석(t 검정) 독립:범주, 종속:연속=============================================')
print('10, 20번 부서 간 평균 연봉 차이 여부를 검정 - 귀무 : 두 부서 간 연봉 평균에 차이가 없다.')
df_10 = pd.read_sql('select buser_num, jikwon_pay from jikwon where buser_num=10', conn)
df_20 = pd.read_sql('select buser_num, jikwon_pay from jikwon where buser_num=20', conn)
print(df_10.head(2))
print(df_20.head(2))
buser10 = df_10['jikwon_pay']
buser20 = df_20['jikwon_pay']

t_result = stats.ttest_ind(buser10, buser20)
print(t_result)
# Ttest_indResult(statistic=0.46171866642116377, pvalue=0.6501365184569681)
print(np.mean(buser10), ' ', np.mean(buser20))  # 5414.285714285715   4904.166666666667
# 해석 : p-value=0.6501 > 0.05 이므로 귀무 채택. 
# 두 부서 간 연봉 평균 5414.285와 4904.16은 통계적으로 차이가 없다라고 말할 수 있다.

print('\n\n===세 집단 이상의 평균에 대한 분산분석(ANOVA, f 검정) 독립:범주, 종속:연속==================================')
print('각 부서 간(4개) 평균 연봉 차이 여부를 검정. 귀무 : 각 부서 간 연봉 평균에 차이가 없다')
df3 = pd.read_sql('select * from jikwon', conn)
print(df3.head(2))

group1 = df3[df3['buser_num'] == 10]['jikwon_pay']
print(group1[:2])
group2 = df3[df3['buser_num'] == 20]['jikwon_pay']
group3 = df3[df3['buser_num'] == 30]['jikwon_pay']
group4 = df3[df3['buser_num'] == 40]['jikwon_pay']


# 데이터 분포 시각화
# plot_data = [group1, group2, group3, group4]
# plt.boxplot(plot_data)
# plt.show()

# 일원분산분석 1
f_sta, p_val = stats.f_oneway(group1, group2, group3, group4)
print('\n결과 1. f_sta : {0:.3f}, p_val : {1:.3f}'.format(f_sta, p_val))
# f_sta : 0.415, p_val : 0.744
# 해석 : p_val : 0.744 > 0.05 이므로 귀무 채택.

# 일원분산분석 2
lmodel = ols('jikwon_pay ~ C(buser_num)', data = df3).fit() # 중심 극한 정리
table = sm.stats.anova_lm(lmodel, type=2)
print('결과 2.\n', table)
# 해석 : PR(>F)(p-value) : 0.74397 > 0.05 이므로 귀무 채택

# 사후검정 - 분산분석을 할 때 항상 사후검정을 해야한다.
from statsmodels.stats.multicomp import pairwise_tukeyhsd   # 사후검정용 모듈
result = pairwise_tukeyhsd(df3.jikwon_pay, df3.buser_num)
print('\n사후검정 . \n',result) # 20 - 40의 값이 가장 크지만, 통계적으로 차이가 없다고 판단할 수 있다.
result.plot_simultaneous()
plt.show()























