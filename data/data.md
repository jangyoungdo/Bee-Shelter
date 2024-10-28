구글 드라이브 링크
https://drive.google.com/drive/folders/1ovZsmdDs7xVZLayLpMe8w8nJ2ok1d9YT?usp=drive_link

```
sound/
├── original/ # Low 데이터
│   ├── honeybee
|   │   ├── internal # 양봉통 안에서 채집한 꿀벌 소리
│   |   └── external # 양봉통 외부에서 채집한 꿀벌 소리
|   |
│   └── hornet 
| 
├── produced/ # 가공 데이터
│   ├── smae_len # 전처리 실험을 위해 꺠끗하다고 판단한 소량의 데이터
|   │   ├── hornet  
|   │   └── honeybee 
|   |
│   ├── Noise_cancel # 나름 노이즈를 제거한 데이터 with WavePad
|   │   ├── hornet  
|   │   ├── honeybee
|   │   ├── produced # 노이즈 제거 후, LPF 진행한 파일
|   |       ├── hornet_500  
|   |       └── honeybee_500
|   |
│   └── LPF # LPF 실험용 데이터
|       ├── 500Hz
|       └── 1000Hz
|
└── CSV/                 
    └── MFCC_CSV #최종 학습에 활용된 데이터

```

