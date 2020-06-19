# ***Regressor 로 정량적 예측 모델 비교    (분류 모델 x)
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

adver = pd.read_csv('../testdata/Advertising.csv')
print(adver.head(2))

x = np.array(adver.loc[:, 'tv':'newspaper'])
y = np.array(adver.sales)
print(x[:2])
print(y[:2])


print('\nKNeighborsRegressor --------------------------------------------------')
kmodel = KNeighborsRegressor(n_neighbors=3).fit(x, y)
print(kmodel)
kpred = kmodel.predict(x)
print('kpred(예측값) : ', kpred[:3])
print('kmodel r2(설명력(결정계수)) : ', r2_score(y, kpred))    # 0.968012077694316


print('\nLinearRegression ------------------------------------------------------')
lmodel = LinearRegression().fit(x, y)
print(lmodel)
lpred = lmodel.predict(x)
print('lpred(예측값) : ', lpred[:3])
print('lmodel r2(설명력(결정계수)) : ', r2_score(y, lpred))    # 0.8972106381789522


print('\nRandomForestRegressor -------------------------------------------------')
rmodel = RandomForestRegressor(n_estimators=100, criterion='mse').fit(x, y) # mse(평균제곱오차) : 예측 모델에서 사용
print(rmodel)
rpred = rmodel.predict(x)
print('rpred(예측값) : ', rpred[:3])
print('rmodel r2(설명력(결정계수)) : ', r2_score(y, rpred))    # 0.9974028994496413


print('\nXGBRegressor ----------------------------------------------------------')   # boosting : 에러가 난 부분을 수정하면서 가중치는 주는 방법이다.
xmodel = XGBRegressor(n_estimators=100, criterion='mse').fit(x, y) # mse(평균제곱오차) : 예측 모델에서 사용    
print(xmodel)
xpred = xmodel.predict(x)
print('xpred(예측값) : ', xpred[:3])
print('xmodel r2(설명력(결정계수)) : ', r2_score(y, xpred))    # 0.9974028994496413
























