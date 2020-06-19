# Naive Bayes Classification : 베이즈 정리를 적용한 확률 분류기
# 텍스트 분류에 효과적 - 스팸메일, 게시판 카테고리 등의 분류에 많이 사용됨
# ML에서는 feature가 주어졌을 때 label의 확률을 구하는데 사용
# P(L|feature) = P(feature}L)P(L) / P(feature)

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]    # 차원 확장
print(x, x.shape)   # 2차원이 됨.
y = np.array([1,3,5,7,9])
model = GaussianNB().fit(x, y)
print(model)
pred = model.predict(x) # GaussianNB(priors=None, var_smoothing=1e-09)
print(pred) # [1 3 5 7 9]

# 새 값
new_x = np.array([[0.5], [7.1], [12.0]])
new_pred = model.predict(new_x)
print(new_pred) # [1 9 9]

print('\n---OneHotEncoding (최소 행렬의 일종)----------------------------------------------------------')
x = np.array([1,2,3,4,5])
x = np.eye(x.shape[0])  # np.eye : 정방행렬이 만들어진다.
print(x)
y = np.array([1,3,5,7,9])

model = GaussianNB().fit(x, y)
print(model)
pred = model.predict(x) 
print(pred)

print('\n---OneHotEncoding : OneHotEncoder----------------------------------------------------------')
x = np.array([1,2,3,4,5])
one_hot = OneHotEncoder(categories='auto')
#print(one_hot)
x = x[:, np.newaxis]
x = one_hot.fit_transform(x).toarray()
print(x)



































