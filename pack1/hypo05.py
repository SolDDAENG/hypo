# 실습) 국가전체와 지역에 대한 인종 간 인원수로 독립성 검정 실습
# 두 집단(국가전체 - national, 특정지역 - la)의 인종 간 인원수의 분포가 관련이 있는가?

import numpy as np
import pandas as pd
import scipy.stats as stats

national = pd.DataFrame(["white"] * 100000 + ["hispanic"] * 60000 + ["black"] * 50000 + ["asian"] * 15000 + ["other"] * 35000)
la = pd.DataFrame(["white"] * 600 + ["hispanic"] * 300 + ["black"] * 250 + ["asian"] * 75 + ["other"] * 150)
# print(national)
# print(la)

# 귀무 : 국가 전체와 지역에 대한 인종 간 인원수는 관련이 없다.(독립적이다)
# 대립 : 국가 전체와 지역에 대한 인종 간 인원수는 관련이 있다.(독립적이지 않다)

# 방법1 : 함수 이용
na_table = pd.crosstab(index=national[0], columns="count")
# print(na_table)
la_table = pd.crosstab(index=la[0], columns="count")
# print(la_table)
na_table["count_la"] = la_table["count"]
print(na_table)

chi2 , p, df, _ = stats.chi2_contingency(na_table)
print("chi2, p, df : ", chi2, p, df)
# chi2, p, df : 18.09952    0.0011    4
# 해석 : p(0.0011) < 0.05    귀무 기각

print('------------------')
# 방법2 : p-value 구하기
# 검정통계량 계산식 sum((관측값 - 기대값)^2 / 기대값
observed = la_table     # 관측값
national_ratio = na_table / len(national)
expected = national_ratio * len(la)   # 기대값
print('expected : ', expected)

chi_square_stat = (((observed - expected) ** 2) / expected).sum()
print('chi_squared_stat : ', chi_square_stat)

# p-value    # cdf() : 누적분포함수, pdf() : 확률밀도함수, pmf() : 확률질량함수
# p-value를 구할땐 누적분포함수가 필요.
pv = 1 - stats.chi2.cdf(x=chi_square_stat, df=4)
print('p-value : ', pv)  # 0.00113047 1