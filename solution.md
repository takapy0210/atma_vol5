# atmaCup#5

## モデル

- LightGBM
    - StratifiedKFold:5fold
    - features: 500前後
    - CV: 0.9022
    - LB: 0.8833

- Catboost
    - StratifiedKFold:5fold
    - features: 500前後
    - CV: 0.8939
    - LB: 0.8777

- LightGBM×0.55 + Catboost×0.45
    - LB: 0.8916


## 特徴量

- あまり暖かみのあるものは作っていません...


### spectrum

- t-SNEとUMAPを用いてそれぞれ2次元圧縮したデータ
- 