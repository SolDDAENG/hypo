# RandomForest 분류 분석 : 여러 개의 DecisionTree를 조합(ensemble)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv")
print(df.head(3), df.shape) # (891, 12)
#print(df.isnull().any())
df = df.dropna(subset=['Embarked', 'Cabin', 'Age'])
print(df.head(3))
df_x = df[['Pclass','Age','Sex']]
print(df_x.head(3))

# Scaling
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
df_x.loc[:, 'Sex'] = LabelEncoder().fit_transform(df_x['Sex'])  # female:0, male:1
print(df_x.head(3))
df_y = df['Survived']
print(df_y.head(3))

# Pclass를 first.class, second.class, third.class

import numpy as np
df_x2 = pd.DataFrame(OneHotEncoder() \
                     .fit_transform(df_x['Pclass'].values[:, np.newaxis]).toarray(), \
                     columns=['f_class','s_class','t_class'], \
                     index = df_x.index)
df_x = pd.concat([df_x, df_x2], axis=1)
print(df_x.head(3))

# train / test dataset
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y)

model = RandomForestClassifier(criterion='entropy', n_estimators=100)
fit_model = model.fit(train_x, train_y)

pred = fit_model.predict(test_x)

print('예측값 : ', pred[:5])
print('실제값 : ', np.array(test_y[:5]))

print('분류 정확도 : ', sum(test_y == pred) / len(test_y))   # 분류 정확도 :  0.7608695652173914
from sklearn.metrics import accuracy_score
print('분류 정확도 : ', accuracy_score(test_y, pred)) # 분류 정확도 :  0.7608695652173914































