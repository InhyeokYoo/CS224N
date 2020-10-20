# Matrix Calculus and Backpropagation

1. 뉴런(Neuron) : Input을 통한 단일 Output을 생성하는 계산 단위
가장 많이 사용되는건 Sigmoid와 binary logistic regression

![image](https://user-images.githubusercontent.com/34912004/96443723-040f7800-1248-11eb-8c6a-9cf229fbc1df.png)

뉴런에서 입력 벡터 x가 어떻게 스케일링되고 합해지며 bias에 추가된 다음, 시그모이드로 전달되는지를 알 수 있다.

2. Feed-forward
분류하기 위해 단어들 사이의 상호작용을 포착

NER: 미리 정의해둔 사람, 회사, 장소 등에 해당하는 단어를 문서에서 인식하여 추출하는 기법 ex) 챗봇


3. Maximum Mrgin Objective Function

"Museums in Paris are amazing" 라벨링 된 ture를 S
"Not All museums in Paris"로 라벨링 된 false를 Sc
라고 가정한다면 우리의 목적 함수는 max(S-Sc) 이거나 min(Sc-S)이다.

우리의 목적함수는 minimize j = max(Sc-S, 0)

하지만 우린 true의 점수가 false보다 긍정적 마진 ∆만큼 높길 바란다. 그러므로,

minimize j = max(∆ + Sc - sm, 0)

우린 이 ∆을 1로 스케일 할 수 있고 다른 조정없이 파라미터들이 적응하게 함. 따라서 모든 트레이닝에 최적화하는 함수를 정의한다.

minimize j = max(1 + Sc - sm, 0)

4. Backpropagation

Feed-forward : 자유투
Backpropogation : 던진 공이 어느 지점에 도착하는지 확인하고 위치를 수정하는 것

![image](https://user-images.githubusercontent.com/34912004/96447852-83537a80-124d-11eb-8172-e6410553cc69.png)
W14(1)은 오직 z1(2)와 a1(2)에만 영향을 끼친다.

자신이 영향을 끼치는 곳에서만 영향을 받는다.

5. Gradient checks

역전파 알고리즘을 수동으로 검증하는 기법

6. Regularization
![image](https://user-images.githubusercontent.com/34912004/96455344-0da0dc00-1258-11eb-815f-711c1310ba9e.png)

λ은 정규화 용어가 원래 비용 함수에 비해 어느 정도의 가중치를 가지는가를 제어
λ의 값이 높으면 가중치가 0에 가깝게 설정되고 모델은 train data로부터 의미있는 것을 배우지 못하며 train, validation, test set의 accuracy가 떨어진다.

7. Dropout
![image](https://user-images.githubusercontent.com/34912004/96456137-10500100-1259-11eb-90ab-05453831941a.png)

더 의미있는 정보를 학습하고 overfitting할 가능성이 낮으며 전반적으로 더 높은 성능을 얻는다.

8. Neuron Units
![image](https://user-images.githubusercontent.com/34912004/96460039-c3225e00-125d-11eb-8ae6-71f15b7c84a6.png)
Sigmoid (0~1)

![image](https://user-images.githubusercontent.com/34912004/96460327-0f6d9e00-125e-11eb-831c-35430811e03f.png)
tanh (-1~1)

![image](https://user-images.githubusercontent.com/34912004/96460378-18f70600-125e-11eb-9828-409a5d333b3b.png)
ReLU (max(z,0)

9. Data Preprocessing
데이터에 대한 기본적인 전처리 수행

Mean Subtraction
데이터에 모든 feature에서 그 평균을 빼는 것
데이터 X -= np.mean(X)

Normalization
![image](https://user-images.githubusercontent.com/34912004/96583927-6fc11600-1318-11eb-870e-98c586d60437.png)

데이터의 분포를 비슷한 크기로 스케일링하는 것

Whitening

데이터를 상관관계가 없고 분산을 1로 만드는 것


Momentum Updates

Optimizer 비교

![image](https://user-images.githubusercontent.com/34912004/96590824-71430c00-1321-11eb-9fc8-c592feece477.png)


