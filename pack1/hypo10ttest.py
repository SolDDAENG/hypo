# 집단에 따라 검정 방법 달라짐

# 수면제1 종류를 먹다가 수면제2 종류를 먹었을 때 수면 시간에 변화여부 검정
import numpy as np
from scipy import stats

x1 = np.array([0.7, 0.3, 0.1, -0.3, 0.2])   # 수면제 1
x2 = np.array([1.0, 1.3, 0.3, -0.1, 0.5])   # 수면제 2

# 서로 다른 사람이 수면제를 복용한 경우 : 독립표본 t검정
result = stats.ttest_ind(x1, x2, equal_var=True)
print(result)

if result.pvalue > 0.05:
    print('수면제 종류에 따른 수면 시간에 변화가 없다.(변화가 미미하다.)')
else:
    print('수면제 종류에 따른 수면 시간에 변화가 있다.')

# 한 사람이 수면제를 복용한 경우 : 대응표본 t검정
result = stats.ttest_rel(x1, x2)
print(result)

if result.pvalue > 0.05:
    print('수면제 종류에 따른 수면 시간에 변화가 없다.(변화가 미미하다.)')
else:
    print('수면제 종류에 따른 수면 시간에 변화가 있다.')
