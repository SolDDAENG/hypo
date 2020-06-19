# BMI(체질량 지수) 식을 이용해 무작위 자료를 작성 후 분류 모델에 적용
# BMI = 몸무게 / 키 * 키
# print(75 / (1.7 * 1.7)) # 25.95155709342561

# 파일을 만들었으니 지금은 필요 없다.
'''
import random
random.seed(123)

def calc_bmi(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 23: return 'normal'
    return 'fat'

# print(calc_bmi(170, 25))

# bmi data 생성 후 파일로 저장
fp = open('bmi.csv','w',encoding='utf-8')
fp.write('height,weight,label\n')
# 무작위로 데이터를 생성
cnt = {'thin':0,'normal':0,'fat':0}
for i in range(50000):  # 5만번 - 너무 큰 데이터는 오래 걸린다.
    h = random.randint(150,200) # 키는 150부터 200사이만 5만개 추출
    w = random.randint(35, 100)
    label = calc_bmi(h, w)
    cnt[label] += 1 # 누적용
    fp.write('{0},{1},{2}\n'.format(h,w,label))

fp.close()
print('저장 완료',cnt)
'''

# SVM으로 bmi data 분류
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

tbl = pd.read_csv('bmi.csv')
print(tbl.head(3))    # 정규화, 표준화를 하는 것이 정확도 높을 수 있다.

# w, h에 대해 정규화하기
label = tbl['label']
w = tbl['weight'] / 100 # 0 ~ 1 사이로 정규화
h = tbl['height'] / 200
wh = pd.concat([w, h], axis=1)  # concat : 병합
print(wh.head(3), wh.shape) # (50000, 2)
print(label[:3], label.shape)   # (50000,)

# train / test dataset : 과적합 방지용
data_train, data_test, label_train, label_test = train_test_split(wh, label) 
print(data_train.shape, data_test.shape)    # (37500, 2) (12500, 2)
print(label_train.shape, label_test.shape)  # (37500,) (12500,)

# model
print()
# model = svm.SVC(C=10).fit(data_train, label_train)
# model = svm.LinearSVC(C=10).fit(data_train, label_train)    # SVC에 비해 속도 향상. 옵션 다양
# C : 정확도를 높일 수 있는 요소
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(data_train, label_train)    # 일반적으로 k값은 3 또는 5를 부여한다. (홀수만 부여)


# k겹 교차검증 : 과적화 방지용 - 데이터의 양이 많지 않거나, train-test를 해도 과적합이 있을 경우 사용한다.
from sklearn import model_selection
cross_val = model_selection.cross_val_score(model, data_train, label_train, cv=5)
print('각각(5겹)의 검증 정확도 : ', cross_val)
print('평균(5겹) 검증 정확도 : ', cross_val.mean())


pred = model.predict(data_test)
print('실제값 : ', label_test[:3])
print('예측값 : ', pred[:3])

# 분류 정확도 확인
ac_score = metrics.accuracy_score(label_test, pred) # 실제값, 예측값 입력
print('분류 정확도 : ', ac_score)    # 분류 정확도 :  0.9964
cl_report = metrics.classification_report(label_test, pred)
print('분류 보고서 : ', cl_report)

# 시각화
tbl = pd.read_csv('bmi.csv', index_col = 2)
print(tbl.head(3))

fig = plt.figure()  # 이미지 저장 선언

def scatter_func(lbl, color):
    b = tbl.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)
    
scatter_func('fat','red')
scatter_func('normal','yellow')
scatter_func('thin','blue')
plt.legend()
plt.savefig('bmi_test.png')
plt.show()
















