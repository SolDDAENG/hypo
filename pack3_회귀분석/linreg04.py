# iris dataset으로 선형회귀
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(2))
print(iris.corr())  # 상관관계

print('\n단순 선형회귀모델 1---------------------------------------------------------------------')
# 단순 선형회귀 모델 작성 sepal_length, sepal_width : -0.117570(r) 약한 음의 상관관계 (별로 적합하지 않긴 함)
result = smf.ols(formula = 'sepal_length ~ sepal_width',data=iris).fit()    # 가설 : 꽃받침의 길이가 꽃받침의 넓이에 영향을 준다?
# print(result.summary()) # 독립변수로서의 자격이 매우 의심스럽다. R-squared(설명력) : 0.014
print('결정계수(설명력) : ', result.rsquared)  # 결정계수(설명력) :  0.013822654141080748    설명력이 매우 떨어짐.
print('p-value : ', result.pvalues) # p-value : 모델 전체의 p-value 1.518983e-01


print('\n단순 선형회귀모델 2---------------------------------------------------------------------')
# 단순 선형회귀 모델 작성 sepal_length, petal_length : 0.871754(r) 매우 강한 양의 상관관계 (매우 적절함)
result2 = smf.ols(formula = 'sepal_length ~ petal_length',data=iris).fit()    # 가설 : 꽃받침의 길이가 꽃잎의 길이에 영향을 준다?
# print(result2.summary()) # 독립변수로서의 자격이 매우 충분하다. R-squared(설명력) : 0.760
print('결정계수(설명력) : ', result2.rsquared)  # 결정계수(설명력) :  0.7599546457725151    설명력이 매우 높음.
print('p-value : ', result2.pvalues) # p-value : 모델 전체의 p-value 1.038667e-47
print('실제값 : ', iris.sepal_length[0], ', 예측값 : ', result2.predict()[0]) 
# 실제값 :  5.1 , 예측값 :  4.879094603339241

print('\n단순 선형회귀모델 - 새로운 데이터로 예측-------------------------------------------------------')
# 새로운 데이터(petal_length)로 sepal_length를 예측 - 둘다 연속형 데이터
new_data = pd.DataFrame({'petal_length':[1.4, 2.4, 0.4]})
y_pred = result2.predict(new_data)
print(y_pred) # 꽃받침의 길이 예측값 : [[0    4.879095], [1    5.288017], [2    4.470172]]
 

print('\n다중 선형회귀모델 ----------------------------------------------------------------------')
# 다중 선형회귀 모델 작성
#result3 = smf.ols(formula = 'sepal_length ~ petal_length + petal_width',data=iris).fit()    # 독립변수를 누구 쓰느냐가 정말 중요하다.
col_selected = '+'.join(iris.columns.difference(['sepal_length', 'species']))  # 독립변수 이외의 것들을 뺀다. sepal_length는 종속변수로 사용한다.
print(col_selected) # petal_length+petal_width+sepal_width
formula = 'sepal_length ~ ' + col_selected  # R에서는 'sepal_length ~ .'을 하면 된다.
result3 = smf.ols(formula=formula, data=iris).fit()   
print(result3.summary()) # 


























