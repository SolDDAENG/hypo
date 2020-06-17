# 분류 모델 성능 평가
# ROC curve (Receiver Operating Characteristic curve) : 
# FPR과 TPR을 각각 x, y축으로 놓은 그래프. ROC curve는 X, Y가 둘다 [0, 1]의 범위이고,
# (0, 0) 에서 (1, 1)을 잇는 곡선이다.
# ROC 커브는 그 면적(AUC - Area Under the Curve)이 1에 가까울수록 (즉 왼쪽위 꼭지점에 다가갈수록) 좋은 성능이다.
# 그리고 이 면적은 항상 0.5~1의 범위를 갖는다. (0.5이면 랜덤에 가까운 성능, 1이면 최고의 성능)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# 분류 연습용 샘플 데이터 작성
x, y = make_classification(n_samples=16, n_features=2, n_informative=2, n_redundant=0, random_state=0)
# print(x)    # [[ 2.03418291 -0.38437236] ...
# print(y)    # [0 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1]

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('y_hat : ', y_hat)    # y_hat :  [1 1 0 1 1 0 0 0 1 0 0 0 1 1 0 1]

f_value = model.decision_function(x)    # 결정함수(판결함수)
print('f_value : ', f_value)    # [ 0.37829565  1.6336573  -1.42938156 ...

print()
df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=["f",'yhat','y'])  # (불확실성, 예측값, 실제값)
print(df)

# ROC 커브 : 머신러닝 모델을 평가할 때 쓰인다.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat, labels=[1, 0]))    # [[7 1] [1 7]]

recall = 7 / (7 + 1)    # TP / (TP + FN)   - TPR
fallout = 1 / (1 + 7)   # FP / (FP + TN)   - FPR
print('recall : ', recall)      # recall :  0.875
print('fallout : ', fallout)    # fallout :  0.125

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y, model.decision_function(x))  # (실제값, 결정함수)
print('fpr : ', fpr)
print('tpr : ', tpr)
print('threshold : ', threshold)    # 재현율을 높이기 위한 판단 기준
# threshold :  [ 3.36316277  2.36316277  1.21967832  0.37829565  0.09428499 -0.76588836 -0.92693183 -4.11896895]

import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0,1],[0,1],'k--','random guess')
plt.plot([fallout],[recall],'ro',ms=10)
plt.xlabel('위양성율(fpr:fallout)')
plt.ylabel('재현율(tpr:recall)')
plt.show()












