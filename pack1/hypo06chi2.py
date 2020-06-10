# 이원카이제곱
# 동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로
# 동일한가를 검정하게 된다. 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.

# 동질성 검정 실습1) 교육방법에 따른 교육생들의 만족도 분석 - 동질성 검정 survey_method.csv
import pandas as pd
import scipy.stats as stats

# 귀무 : 교육방법에 따른 교육생들의 만족도에 차이가 없다.
# 연구 : 교육방법에 따른 교육생들의 만족도에 차이가 있다.

data = pd.read_csv('../testdata/survey_method.csv')
print(data.head())
print(data['method'].unique())  # [1 2 3]    # unique : 중복을 제외하고 출력
print(data['survey'].unique())  # [1 2 3 4 5]

ctab = pd.crosstab(index=data['method'], columns=data['survey'])  # 교육 방법별 만족도
ctab.columns = ['매우 만족', '만족', '보통', '불만족', '매우 불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)
chi2, p, df, _ = stats.chi2_contingency(ctab)  # chi2_contingency() 명령의 결과는 튜플로 반환되며 첫번째 값이 검정통계량, 두번째 값이 유의확률
print("chi2;{}, p:{}, df:{}".format(chi2, p, df))  # df : 자유도
# chi2;6.544667820529891, p:0.5864574374550608, df:8
# 해석 : p(0.5864) > 0.05 --> 귀무 채택 : 교육방법에 따른 교육생들의 만족도에 차이가 없다.

print('**' * 30)
# 동질성 검정 실습2) 연령대별 sns 이용률의 동질성 검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보
# 전략을 세우고자 한다.
# 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자

# 귀무 : 연령대별로 SNS 서비스틀에 이용 현환이 서로 동일하다.
# 연구 : 연령대별로 SNS 서비스틀에 이용 현환이 서로 동일하지 않다.

snsdata = pd.read_csv('../testdata/snsbyage.csv')
print(snsdata.head())
print(snsdata['age'].unique())  # [1 2 3]    # 20대, 30대, 40대
print(snsdata['service'].unique())  # ['F' 'T' 'K' 'C' 'E']    # SNS 서비스사
print()

sns_ctab = pd.crosstab(index=snsdata['age'], columns=snsdata['service'])  # crosstab : crosstable : 빈도표 만들기
print(sns_ctab)

chi2, p, df, _ = stats.chi2_contingency(sns_ctab)
print('chi2:{}, p:{}, df:{}'.format(chi2, p, df))
# chi2:102.75202494484225, p:1.1679064204212775e-18, df:8
# 해석 : p:1.1679064204212775e-18 < 0.05 (p가 거의 0수준...) : 귀무가설 기각
