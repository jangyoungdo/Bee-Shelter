## 왜 주파수인가?
- 시계열 정보의 진폭(Magnitude)의 경우는 데이터 수집 단계의 환경 변수에 너무 취약하다.
- 내가 애초에 DAQ 장비를 활용한 것이 아니라 녹음기로 실험을 진행한 부분이고, 수집 환경 또한 양봉장과 박물관 같은 노이즈에 아주 취약한 환경임
- 물론 더욱 일반적인 모델을 만들 수 있을 지 모르겠지만, 일단 모델을 만드는 단계에서 나는 특징값에 더욱 주목했고, 때문에 주파수 영역에서의 다양한 시도를 진행했음

---

## 디바이스와 합치기 이전에 모델 및 데이터에 대해 이해하는 것을 다시 시작
- 주파수와 샘플 레이트에 대한 더 높은 이해 필요
- 또한 염가형 마이크로폰의 경우 샘플레이트가 굉장히 낮기에 2000으로 재지정 필요
- LPF(Low Pass Filter) 1000Hz로 했을 때보다 500Hz로 했을 때, 비지도 학습 Clustering 정도가 높은 것을 확인할 수 있었음
- clustering이 잘 진행 되었을 때, Logistic 회귀 진행이 잘 되는 것을 확인할 수 있었음


## 전처리 과정을 통한 특징값 검증

|MFCC| Low Pass Filter(1000Hz) | Low Pass Filter(500Hz) |
|:------------------:|:------------------:|:------------------:|
|꿀벌| ![image](https://github.com/user-attachments/assets/5d4f51dc-f5f1-42b9-a288-299d9651ebfe) | ![image](https://github.com/user-attachments/assets/dd5fc1e1-44c7-4bc1-9714-4c97c30f21d2) |
|말벌| ![image](https://github.com/user-attachments/assets/74769773-7b52-4d33-b4ff-ea5827c27076) | ![image](https://github.com/user-attachments/assets/a08c194f-a938-410f-b664-c67159f8a13e) |
|Clustering <br> Accuracy| ![image](https://github.com/user-attachments/assets/f4a46471-14fd-42a5-8099-d3e994a80702) <br> ![image](https://github.com/user-attachments/assets/28cec96b-8a00-4576-b3d2-0da834a5d24d) | ![image](https://github.com/user-attachments/assets/a20eb4ef-73bb-4632-84e2-d6ae5f074600) <br> ![image](https://github.com/user-attachments/assets/455fc19e-f318-4811-a440-bcee1333351d) |
| Logistic <br> Regression <br> Accuracy| ![image](https://github.com/user-attachments/assets/402c84dd-a33e-4cce-be21-22f52155f9fb) | ![image](https://github.com/user-attachments/assets/9b10d167-dc56-4a64-a5d4-c0eb8e920f66) |



- 일전에 필터를 적용했을 때, 배음이 갖는 특징을 제거할 수 있다는 의혹이 있었으나 오히려 비지도 학습 Clustering을 진행했을 때, 오히려 Accuracy가 증가하는 것으로 보아 노이즈로 인한 방해요소가 더 큰 것으로 보여짐
- 배음 ⇒ 하나의 음을 구성하는 여러 부분음들 중, 기본음(基本⾳)보다 높은 정수배의 진동수를 갖는 모든 상음(上⾳)들을 가리키는 말이다.

- 유튜버들의 경우 답변이 없어 데이터 추가수급은 어려울 것으로 판단 ⇒ 추후에 또 답변을 보내던가 할 듯, But 말벌 활동기가 끝남
- 현재 가지고 있는 데이터를 편집해서 데이터량을 늘리는 방향을 선택
- WavePad를 가지고 아주 이상한 부분 편집
- 꿀벌(안) ⇒ 1시간 분량
- 말벌 ⇒ 2시간 분량스마트 양봉통을 위한 꿀벌, 말벌 분류 모델 
  그리고 소량으로 지도 학습을 진행한 후, 수도 레이블링을 진행(준비도 학습) ⇒ 과적합을 방지하고 정확도를 좀 더 향상시켜볼 것

---

## Pseudo - Labeling 진행

- 위의 Clustering으로 성능을 확인한 전처리 방법을 그대로 적용할 것
- 모델은 최대한 심플한(?), 가장 특징값에 포커싱할 수 있도록 Logistic Regression으로 진행할 것
- 이후 과적합 예방과 보다 일반적인 모델을 제작하기 위해 준지도 학습인 Pseudo - Labeling을 진행
- n_samples
  1. n_samples의 양을 6000으로 처음 진행 ⇒ 약 83%의 정확도 확인
  2. n_samples의 양을 늘려보기도 하고 줄여보기도 함 ⇒ 낮출 수록 정확도가 조금씩 높아짐
  3. n_samples의 양을 100부터 늘려가며 그래프로 나타내봄 ⇒ 100일 때, 가장 높음(Accuracy : 99.59 % )
 
![image](https://github.com/user-attachments/assets/bece8fa2-2393-4c17-8f2e-37828697aa2f)

- threshold
  1. Threshold의 정도를 변화했을 떄, 어떤 변화가 발생할지 확인
  2. 신뢰도에 따라 수도 라벨링에 활용되는 데이터 양이 확 줄어들면 진행하는 의미가 없음
  3. 때문에 Threshold에 따른 활용되는 데이터 양 추이 확인

![image](https://github.com/user-attachments/assets/081fcd6e-7a2d-46a0-a72e-0684343396bc)

- 신뢰구간 + 샘플 량을 변화하며 가장 성능이 높은 모델 찾기

![image](https://github.com/user-attachments/assets/bcafabe8-3c5a-45b5-8510-9cc2819eecaf)
![image](https://github.com/user-attachments/assets/0e99dbe4-a596-4991-ac08-e42c0cd751ab)

  => 큰 차이를 보이지는 않으나 n_samples = 100, Threshold = 0.95일 때, 성능이 가장 높음을 확인할 수 있음

---

## 모델에서 가장 높은 가중치 확인 및 검증
- n_samples를 100, 신뢰 구간을 0.95로 세팅했을 때의 모델의 성능으로 독립변수들의 가중치를 구해봄
- 모델 제작에서 로지스틱 회귀 모델 수식의 가중치를 확인
- librosa.mfcc를 역추산하여 가장 높은 가중치를 갖는 Feature가 나타내는 주파수 대역 확인

 ![image](https://github.com/user-attachments/assets/29588abe-92de-4a1a-89f8-904bc5fea090)
 ![image](https://github.com/user-attachments/assets/56fe3797-91ba-4579-898c-f141569145f1) ![image](https://github.com/user-attachments/assets/c2e1d067-f451-42f9-8c46-9f578d8b0574)


- 그 결과 말벌의 고유 주파수인 약 100Hz가 포함된 MFCC4의 가중치가 가장 높은 것을 확인할 수 있었음
- 그치만 그 가중치를 확인해 보면 -0.226459임을 확인할 수 있는데, 이 말은 즉슨 해당 Feature값이 커질수록 말벌이 아닌 꿀벌로 판단할 확률이 올라간다는 것을 의미함.
- 혹은 선형 문제에서 빈번한 다중공선성에 의거한 오류일 수 있음
- 특정 변수가 다른 변수에 의해 설명되는 정도가 커져, 그 변수의 기여가 부정적(음수)으로 나타날 수 있음

- Correlation Matrix
  
![image](https://github.com/user-attachments/assets/9a0835a9-804e-40b3-a61c-c4ad4fedff46)

- VIF 검사(다중공선성 검)
  
![image](https://github.com/user-attachments/assets/5a3caa06-79c8-4673-82ef-ea3554baf1f2)

- 말벌의 고유 주파수와 가까운 주파수 대역의 MFCC값에서 VIF 값이 높게 측정됨
- 다중공선성의 가능성이 매우 높은 것을 확인할 수 있었음
- 그래서 뽑은 특징값들에 대해 다시 한번 PCA를 진행
- 다중공선성을 어느 정도 예방하고자 함
- 만약 그 영향이 있었다면 어느 정도 성능 향상이 보일 것으로 생각됨

![image](https://github.com/user-attachments/assets/c7eaa458-652d-4022-abfd-4ea85a2de906) 

![image](https://github.com/user-attachments/assets/f5ac1274-2bd1-45b8-9479-3bb7fed71d89)

![image](https://github.com/user-attachments/assets/2c0d5a46-9483-4f10-b9dc-7ebc91373019)





