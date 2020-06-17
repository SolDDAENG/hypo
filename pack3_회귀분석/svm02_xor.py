# SVM으로 XOR 분류

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

xor_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

xor_df = pd.DataFrame(xor_data)
print(xor_df)
feature = np.array(xor_df.iloc[:, 0:2])
label = np.array(xor_df.iloc[:,2])
print('feature : \n', feature)
print('label : ', label)

# model = LogisticRegression()    # 선형분류로는 xor을 구할 수 없다.
model = svm.SVC()   # SVC에서 알아서 커널트릭을 이용해서 label을 고차원으로 만들어준다.
model.fit(feature, label)   # 2차원, 1차원. => SVC가 고차원으로 올려준다.

pred = model.predict(feature)
print('pred : ', pred)

# 분류 리포트
acc = metrics.accuracy_score(label, pred)   # 분류 정확도 : accuracy_score(실제값, 예측값)
print('분류 정확도 : ', acc) # 분류 정확도 :  1.0 = 100%
ac_report = metrics.classification_report(label, pred)
print(ac_report)























