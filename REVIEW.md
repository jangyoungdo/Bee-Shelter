## 왜 주파수인가?
- 시계열 정보의 진폭(Magnitude)의 경우는 데이터 수집 단계의 환경 변수에 너무 취약하다.
- 내가 애초에 DAQ 장비를 활용한 것이 아니라 녹음기로 실험을 진행한 부분이고, 수집 환경 또한 양봉장과 박물관 같은 노이즈에 아주 취약한 환경임
- 물론 더욱 일반적인 모델을 만들 수 있을 지 모르겠지만, 일단 모델을 만드는 단계에서 나는 특징값에 더욱 주목했고, 때문에 주파수 영역에서의 다양한 시도를 진행했음

## 디바이스와 합치기 이전에 모델 및 데이터에 대해 이해하는 것을 다시 시작
- 주파수와 샘플 레이트에 대한 더 높은 이해 필요
- 또한 염가형 마이크로폰의 경우 샘플레이트가 굉장히 낮기에 2000으로 재지정 필요
- LPF(Low Pass Filter) 1000Hz로 했을 때보다 500Hz로 했을 때, 비지도 학습 Clustering 정도가 높은 것을 확인할 수 있었음
- clustering이 잘 진행 되었을 때, Logistic 회귀 진행이 잘 되는 것을 확인할 수 있었음


## 전처리 과정을 통한 특징값 검증

|MFCC| Low Pass Filter(1000Hz | Low Pass Filter(500Hz) |
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

  1. n_samples의 양을 6000으로 처음 진행 ⇒ 약 83%의 정확도 확인
  2. n_samples의 양을 늘려보기도 하고 줄여보기도 함 ⇒ 낮출 수록 정확도가 조금씩 높아짐
  3. n_samples의 양을 100부터 늘려가며 그래프로 나타내봄 ⇒ 100일 때, 가장 높음(Accuracy : 99.59 % )



