# 미국, 일본, 중국 사람들의 한국 관광지 선호 지역 상관관계 분석
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rc('font', family='malgun gothic')


def setScatterCorr(tour_table, all_table, tourpoint):
    #     print(tourpoint)    # 창덕궁 운현궁 경복궁 창경궁 종묘 순서대로 처리
    # 계산할 관광지명에 해당하는 데이터만 뽑아 tour에 저장하고, 외국인 관광객 자료와 병합
    tour = tour_table[tour_table['resNm'] == tourpoint]
    #     print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    #     print(merge_table)

    # 시각화
    plt.subplot(1, 3, 1)
    plt.xlabel('중국인 입장수')
    plt.ylabel('외국인 입장객수')
    # 상관계수 r 얻기
    lamb1 = lambda p: merge_table['China'].corr(merge_table['ForNum'])  # corr : 상관계수 함수
    r1 = lamb1(merge_table)
    plt.title('r : {:.5f}'.format(r1))  # 상관계수를 제목에 표시
    plt.scatter(merge_table['China'], merge_table['ForNum'], s=6, c='black')  # s: size, c = color

    plt.subplot(1, 3, 2)
    plt.xlabel('일본인 입장수')
    plt.ylabel('외국인 입장객수')
    # 상관계수 r 얻기
    lamb2 = lambda p: merge_table.Japan.corr(merge_table['ForNum'])  # corr : 상관계수 함수    #
    r2 = lamb2(merge_table)
    plt.title('r : {:.5f}'.format(r2))  # 상관계수를 제목에 표시
    plt.scatter(merge_table.Japan, merge_table['ForNum'], s=6, c='blue')  # s: size, c = color
    #     merge_table['China'], merge_table.Japan 둘다 같다. 어떻게 쓰던 상관없다.

    plt.subplot(1, 3, 3)
    plt.xlabel('미국인 입장수')
    plt.ylabel('외국인 입장객수')
    # 상관계수 r 얻기
    lamb3 = lambda p: merge_table['USA'].corr(merge_table['ForNum'])  # corr : 상관계수 함수    #
    r3 = lamb3(merge_table)
    plt.title('r : {:.5f}'.format(r3))  # 상관계수를 제목에 표시
    plt.scatter(merge_table['USA'], merge_table['ForNum'], s=6, c='red')  # s: size, c = color

    plt.tight_layout()  # 간격을 적당하게 알아서 준다.
    plt.show()

    return [tourpoint, r1, r2, r3]  # r1, r2, r3 => 상관계수


def Gogo():
    fname = '서울특별시_관광지입장정보_2011_2016.json'
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    # print(jsonTP, type(jsonTP))   # <class 'list'>
    tour_table = pd.DataFrame(jsonTP, columns=['yyyymm', 'resNm', 'ForNum'])  # '년월일', '관광지명','입장객수'만 출력
    tour_table = tour_table.set_index('yyyymm')
    # print(tour_table.head(10))  # 201101  창덕궁   14137

    # 관광지 이름 얻기
    resNm = tour_table.resNm.unique()
    # print('관광지 이름 : ', resNm, len(resNm), type(resNm))  # 16개 <class 'numpy.ndarray'>
    # ['창덕궁' '운현궁' '경복궁' ... '롯데월드']
    print('대상 관광지 이름 : ', resNm[:5])  # ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘'] 5개만 처리할 예정

    # 중국인 관광객 정보
    cdf = '중국인방문객.json'
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())
    # print(jdata)
    china_table = pd.DataFrame(jdata, columns=['yyyymm', 'visit_cnt'])  # '년월일', '방문자수'만 출력
    china_table = china_table.rename(columns={'visit_cnt': 'China'})  # 컬럼의 이름 변경
    china_table = china_table.set_index('yyyymm')  # index에 '년월일'을 준다.
    print(china_table.head(3))

    # 일본인 관광객 정보
    jdf = '일본인방문객.json'
    jdata = json.loads(open(jdf, 'r', encoding='utf-8').read())
    # print(jdata)
    japan_table = pd.DataFrame(jdata, columns=['yyyymm', 'visit_cnt'])  # '년월일', '방문자수'만 출력
    japan_table = japan_table.rename(columns={'visit_cnt': 'Japan'})  # 컬럼의 이름 변경
    japan_table = japan_table.set_index('yyyymm')  # index에 '년월일'을 준다.
    print(japan_table.head(3))

    # 미국인 관광객 정보
    udf = '미국인방문객.json'
    jdata = json.loads(open(udf, 'r', encoding='utf-8').read())
    # print(jdata)
    usa_table = pd.DataFrame(jdata, columns=['yyyymm', 'visit_cnt'])  # '년월일', '방문자수'만 출력
    usa_table = usa_table.rename(columns={'visit_cnt': 'USA'})  # 컬럼의 이름 변경
    usa_table = usa_table.set_index('yyyymm')  # index에 '년월일'을 준다.
    print(usa_table.head(3))

    # merge (병합)
    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)  # 중국과 일본 테이블 병합
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)  # all과 미국 테이블 병합
    # print(all_table.head(3))    # 201101   91252  209184  43065

    r_list = []  # 각 관광지(5군대) 마다 상관계수를 구해 기억

    for tourpoint in resNm[:5]:
        # print(tourpoint)
        # 시각화 + 상관계수 처리 함수를 호출
        r_list.append(setScatterCorr(tour_table, all_table, tourpoint))

    # print(r_list)
    r_df = pd.DataFrame(r_list, columns=('고궁명', '중국', '일본', '미국'))
    r_df = r_df.set_index('고궁명')
    print(r_df)

    r_df.plot(kind='bar', rot=50)
    plt.show()


if __name__ == '__main__':
    Gogo()
