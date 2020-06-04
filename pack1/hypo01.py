# hypothesis : 추론
# 분산 / 표준편차의 중요성 - 데이터의 치우침을 표현하는 대표적인 값 중 하나

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
print(stats.norm(loc=1, scale=2).rvs(10))  # loc : 기대값, scale : 표준편차, rvs(10) : 난수(10개)
# 기대값이 1이고 표준편차가 2인 난수 10개 출력
# 기대값 : 어떤 확률을 가진 사건을 무한 반복할 때 얻을 수 있는 값의 평균. 기대할 수 있는 값.

print()
centers = [1, 1.5, 2]
col = 'rgb'

std = 0.1  # 표준편차 = 0.1
data_1 = []  # data_1의 값은 리스트에 담는다.
for i in range(3):  # range : 입력받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 만들어 리턴.
    data_1.append(stats.norm(loc=centers[i], scale=std).rvs(100))  # list.append(x) : 리스트 끝에 x개를 넣는다.
    # print(data_1)
    plt.plot(np.arange(len(data_1[i])) + i * len(data_1[0]), data_1[i], '*', color=col[i])

# np.arange(len(data_1[i])) + i * len(data_1[0]) : x값, data_1[i] : y값
# arange : 일정하게 떨어져 있는 숫자들을 array 형태로 변환

plt.show()

std = 2  # 표준편차 = 2
data_1 = []  # data_1의 값을 리스트에 담는다.
for i in range(3):  # range : 입력받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 만들어 리턴.
    data_1.append(stats.norm(loc=centers[i], scale=std).rvs(100))
    # print(data_1)
    plt.plot(np.arange(len(data_1[i])) + i * len(data_1[0]), data_1[i], '*', color=col[i])
# np.arange(len(data_1[i])) + i * len(data_1[0]) : x값, data_1[i] : y값

plt.show()