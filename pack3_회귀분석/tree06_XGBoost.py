# http://cafe.daum.net/flowlife/RUrO/93
# RandomForest vs xgboost
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import xgboost as xgb       # pip install xgboost   

if __name__ == '__main__':
    iris = datasets.load_iris()
    print('아이리스 종류 :', iris.target_names)
    print('데이터 열 이름 :', iris.feature_names)

    # iris data로 Dataframe
    data = pd.DataFrame(
        {
            'sepal length': iris.data[:, 0],
            'sepal width': iris.data[:, 1],
            'petal length': iris.data[:, 2],
            'petal width': iris.data[:, 3],
            'species': iris.target
        }
    )
    print(data.head(2))

    x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
    y = data['species']


    # 테스트 데이터 30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    # 학습 진행
    #model = RandomForestClassifier(n_estimators=100)  # RandomForestClassifier
    model = xgb.XGBClassifier(booster='gbtree',           # XGBClassifier
                    max_depth=4, 
                    n_estimators=100) 

    # 속성 - booster: 의사결정 기반 모형(gbtree), 선형 모형(linear)
    #    - max_depth [기본값: 6]: 과적합 방지를 위해서 사용되며 CV를 사용해서 적절한 값이 제시되어야 하고 보통 3-10 사이 값이 적용된다.

    model.fit(x_train, y_train)


    # 예측
    y_pred = model.predict(x_test)
    print('예측값 : ', y_pred[:5])
    print('실제값 : ', np.array(y_test[:5]))
    print('정확도 : ', metrics.accuracy_score(y_test, y_pred))
    
    
    
    
    
    
    