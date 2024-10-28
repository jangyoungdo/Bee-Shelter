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

|| Low Pass Filter(1000Hz | Low Pass Filter(500Hz) |
|:------------------:|:------------------:|:------------------:|
|MFCC(꿀벌)| ![image](https://github.com/user-attachments/assets/5d4f51dc-f5f1-42b9-a288-299d9651ebfe) | ![image](https://github.com/user-attachments/assets/dd5fc1e1-44c7-4bc1-9714-4c97c30f21d2) |


|Low Pass Filter(1000Hz)|Low Pass Filter(500Hz)|

  - MFCC(꿀벌)
    ![image](https://github.com/user-attachments/assets/5d4f51dc-f5f1-42b9-a288-299d9651ebfe)

  - MFCC(말벌)
    ![image](https://github.com/user-attachments/assets/74769773-7b52-4d33-b4ff-ea5827c27076)

  - Clustering Accuracy
    
    ![image](https://github.com/user-attachments/assets/f4a46471-14fd-42a5-8099-d3e994a80702)
    ![image](https://github.com/user-attachments/assets/28cec96b-8a00-4576-b3d2-0da834a5d24d)

  - Logistic Regression Accuracy
    
    ![image](https://github.com/user-attachments/assets/4901a280-2642-4419-aae4-9760f1d23434)

- Low Pass Filter(500Hz)
    
  - MFCC(꿀벌)
    ![image](https://github.com/user-attachments/assets/dd5fc1e1-44c7-4bc1-9714-4c97c30f21d2)

  - MFCC(말벌)
    ![image](https://github.com/user-attachments/assets/a08c194f-a938-410f-b664-c67159f8a13e)



    
