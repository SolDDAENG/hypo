# SVM 으로 얼굴 이미지 분류
from sklearn.datasets import fetch_lfw_people   # 정치인 얼굴사진 지원
import matplotlib.pyplot as plt
from sklearn.metrics._classification import classification_report

faces = fetch_lfw_people(min_faces_per_person=60)   # min_faces_per_person : 사진 개수 선택, color : True는 컬러. default는 False
# print(faces)    # {'data': array([[171., 146., ...
print(faces.target_names)   # ['Ariel Sharon' 'Colin Powell' ...
print(faces.images.shape)   # (1348, 62, 47, 3)

fig, ax = plt.subplots(3, 5)
print(fig)  # Figure(640x480)
print(ax.flat, ' ', len(ax.flat))   # 15

for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])   # x, y의 좌표값 없애고, 얼굴을 차례대로 출력
plt.show()

# PCA : 이미지 전체가 아니라 주요 특징만을 추출한 이미지로 분류 작업을 수행
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline  # 작업을 묶어서 실행하기 위해 사용

m_pca = PCA(n_components=150, whiten=True, random_state=0)
m_svc = SVC()
model = make_pipeline(m_pca, m_svc)
print(model)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=1)
print(x_train.shape)    # (1011, 2914)
print(y_train.shape)    # (1011,)

model.fit(x_train, y_train)

yfit = model.predict(x_test)    # 예측값
print('예측값 : ', yfit)   #  [1 4 1 3 3 3 7 3 ...

# 분류 보고서
from sklearn.metrics import classification_report
print(classification_report(y_test, yfit, target_names=faces.target_names))

# 분류 정확도
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
mat = confusion_matrix(y_test, yfit)
print(mat)
print('분류 정확도 : ', accuracy_score(y_test, yfit))    # 분류 정확도 :  0.7952522255192879

# test 이미지 중 일부 자료로 예측한 값과 비교를 위한 시각화
# plt.imshow(x_test[0].reshape(62, 47), cmap='bone')
# plt.show()

fig, ax = plt.subplots(4, 6)
# print(ax)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == y_test[i] else 'red')
    fig.suptitle('pred', size=15)
plt.show()

# 클래스들 간에 오차행렬 시각화
import seaborn as sns

sns.heatmap(mat.T, square=True, annot=True, cbar=False,
            xticklabels=faces.target_names, 
            yticklabels=faces.target_names) 
plt.xlabel('true label')    # 실제값
plt.ylabel('predicted label')   # 예측값
plt.show()





















