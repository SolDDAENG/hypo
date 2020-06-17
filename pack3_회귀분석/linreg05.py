# mtcars dataset으로 선형회귀분석
# 귀납적/연역적 추론. 통계학은 귀납적 추론(개별 사례를 모아 일반적인 법칙(모델)을 생성) 
# 귀납 : 개별 사례들을 모아 일반적인 법칙을 만드는 것.

import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data  # 웬만한 데이터는 kaggle에 다 있다.

print(mtcars)  # [32 rows x 11 columns]
print(mtcars.describe())
print(mtcars.info)

# print(mtcars.corr())    # 상관관계 확인
print(np.corrcoef(mtcars.hp, mtcars.mpg))  # 마력수, 연비 상관관계    # 마력수가 올라갈수록 연비가 떨어진다. -> 음의상관관계(반비례관계)

# 시각화
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수')
# plt.ylabel('연비')
# slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1) # R의 abline과 같은 효과
# print(mtcars.hp * slope + intercept)
# plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
# plt.show()

print('\n---단순 선형회귀 분석--------------')  # 독립변수가 하나
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()  # 이 데이터를 가지고 학습을 한다.
# ols = 최소자승법 -  어떤 데이터가 주어졌을 때 최적의 추세선을 그리기 위한 방법 중 하나
# print(result.conf_int(alpha=0.05))  # 95% 신뢰구간값
# print(result.conf_int(alpha=0.01))  # 99% 신뢰구간값
# print(result.summary())
print(result.summary().tables[1])
# yhat = -0.0682 * 110 + 30.0989
print('예측 연비 : ', -0.0682 * 110 + 30.0989)  # 22.5969    # 마력수가 높아지만 연비가 낮아지고
print('예측 연비 : ', -0.0682 * 50 + 30.0989)  # 26.6889    # 마력수가 낮아지면 연비가 높아진다.
print('예측 연비 : ', -0.0682 * 200 + 30.0989)  # 16.4589
# msg(연비)sms hp(마력수) 값의 -0.0682 배 씩 영향을 받고 있다.
# 마력에 따라 연비는 증감한다. 라고 말할 수 있으나 이는 조심스럽다. 왜냐하면 일반적으로 독립변수는 복수이기 때문이다.....
# 모델이 제공한 값을 믿고 섣불리 판단하는 것은 곤란하다. 의사결정을 위한 참고 자료로 사용해야 한다.

print('\n---다중 선형회귀 분석 (독립변수 복수)-------')
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())
# Adj. R-squared(설명력):                  0.815

print('\n---추정치 구하기 : 임의의 차체무게에 대한 연비 출력----------------|')
print(np.corrcoef(mtcars.wt, mtcars.mpg))  # -0.86765938 => 음의 상관관계가 매우 강하다.
result3 = smf.ols(formula='mpg ~ wt', data=mtcars).fit()  # wt : 차체 무게
print(result3.summary())
# 결정계수(R-squared) : 0.753
# 모델의 p-value(Prob (F-statistic)) : 1.29e-10 

# 추정치(예측값) 출력
kbs = result3.predict()
# print('에측값 : ', kbs[:3])
# print('실제값 : ', mtcars.mpg[:3])

data = {
    'mpg':mtcars.mpg,
    'mpg_pred':kbs,
}
df = pd.DataFrame(data)
print(df)  # 실제 연비와 추정 연비가 대체적으로 비슷한 것을 알 수 있다.

print()
# 이제 새로운 데이터(차체 무게)로 연비를 추정
mtcars.wt = 6  # 차체의 무게가 6톤이라면 연비는?
ytn = result3.predict(pd.DataFrame(mtcars.wt))  # predict : 예측값 출력
print('차체 무게가 6톤이라면 연비는??', ytn[0])  # 5.128

mtcars.wt = 0.4  # 차체의 무게가 400kg 이라면 연비는?
ytn = result3.predict(pd.DataFrame(mtcars.wt))
print('차체 무게가 6톤이라면 연비는??', ytn[0])  # 35.14

# 복수 차체무게에 대한 연비 예측
wt_new = pd.DataFrame({'wt':[6, 3, 1, 0.4, 0.3]})
pred_mpgs = result3.predict(wt_new)
print('예상 연비 : ', np.round(pred_mpgs.values, 3))  # [ 5.218 21.252 31.941 35.147 35.682]    
# np.round(pred_mpgs.values 값을, 반올림해서 소수섬 '3'째 자리까지 출력)

