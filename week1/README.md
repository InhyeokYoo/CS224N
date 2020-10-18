# Week 1 (10.13.20)

발표자: 유인혁

1주차이다 보니 간단한 내용이라서 설명 위주보단, issue 위주로 내용을 구성하는 것이 맞아보인다.

# 1. Introduction and Word Vectors

Traditional한 NLP(i.e. TF-IDF라던가 neural net에 기반하지 않는 method)를 지나, distributed representation과 같은 기초개념부터 소개하고 있다. 가장 처음 보는 것은 Mikolov의 word2vec이다.

Word2vec은 Skip-gram 위주로 설명이 되어 있다. 일반적으로는 CBOW보단 Skip-gram이 더 성능이 좋은 것으로 알려져 있는데, 아무래도 빈번한 업데이트가 원인으로 추정된다.
[다음](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)은 스터디에서 언급했던 *Neural Word Embeddingas Implicit Matrix Factorization (Levy and Goldberg, 2014)*로, SGNG(Skip Gram with Negative Sampling)을 matrix-factorization 관점에서 소개한 논문이다 (NIPS 2014). *Shifted Positive* PMI와 SVD를 사용하면 word2vec과 동일한 embedding을 얻을 수 있다고 한다.

word2vec의 결과로 얻어지는 embedding은 U, V 두 개가 있다. 보통 사용하는 것은 U matrix이다. 앞서 언급한 Goldberg와 Levy의 [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)에서는 context word를 계산하는 V와 center word U를 서로 다른 vocabulary로 간주한다. 다음은 이에 대한 전문이다.

> Throughout  this  note,  we  assume  that  the  words  and  the  contexts  come  from distinct vocabularies, so that, for example, the vector associated with the word *dog* will be different from the vector associated with the context *dog*.  This assumption follows the literature, where it is not motivated.  One motivation for making this assumption is the following:  consider the case where both the word *dog* and the context *dog* share the same vector v.  Words hardly appear in  the  contexts  of  themselves,  and  so  the  model  should assign  a  low  probability  to p(dog|dog), which entails assigning a low value to v·v which is impossible.

## Issue

다음은 스터디에서 명확하게 밝히지 못한 내용이다.

- Negative sampling
    - Negative sampling은 candidate sampling의 일종으로서, softmax의 연산을 줄여주는 역할을 한다. 다만, 최신 논문에선 못 본것 같다....
    - 다만 추천시스템에선 많이 쓰는 듯하다.
        - [Candidate sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf)
    - Noise Constrative Estimation (NCE)의 일종으로 알고 있는데, 정확한 관계는 모르겠다.
        - **softmax연산을 logistic으로 어떻게 근사하는지 알 필요가 있다.**
        - > 우선, w와 c를 다음과 같이 정하자.  
w: 중심 단어의 Vector Representation (또는 Feature Vector 또는 Embedding 또는 Weights)  
c: 주변 단어의 Vector Representation (또는... 상동)  
그리고 문서에서 단어를 이동하면서 **중심 단어 w와 context 단어 c의 쌍**을 만들고 이것을 Good 쌍이라고 하자.  
Good 쌍: (w, c)  
그리고 현재의 중심 단어 w의 현재의 Window 내에 **있지 않는** 다른 어떤 단어 n의 쌍을 만들고 이것을 Bad 쌍이라고 하자.   
Bad 쌍: (w, n)  
그리고 다음과 같은 목표로 학습시킨다.  
sigmoid(w · c) = 1   
sigmoid(w · n) = 0  
잘 알다시피 sigmoid는 0과 1 사이의 값을 내는 함수다. 그리고 sigmoid(z)는 z 값이 클수록 (1을 향해) 커지고 z 값이 작을수록 (0을 향해) 작아진다.
sigmoid(w · c)를 1을 target으로 학습한다는 것은 w와 c를 같은 방향을 가르키는 벡터로 만들겠다는 뜻이고, 0을 target으로 학습한다는 것은 w와 n을 다른 방향을 가르키는 벡터로 만들겠다는 뜻이다.
그런데 여기서 c와 n은 사실 하나의 Vector Representation이라 할 수 있다. 같은 단어 (ex: 콜라) 하나가 어떤 때는 c가 되고 또 다른 때는 n이 되는 것이기 때문에 그렇다. 그래서 n도 c로 쓸 수 있다. 그러니까 정리하면 Word2Vec은 w와 c 쌍에 대해서 어떤 때는 sigmoid(w · c) = 1, 또 어떤 때는 sigmoid(w · c) = 0 이렇게 학습 시키는 것이다. Skip-Gram이 Neural Network이다 보니 NN으로 많이 그린다. 이때 w는 입력 레이어와 가운데 레이어 사이의 weights이고, c는 가운데 레이어와 출력 레이어 사이의 weights이기도 하다. 그런데 사실 NN으로 생각하지 않고 위와 같이 두 Vector의 내적으로 생각하는 것이 더 쉽고 명확하게 이해 된다. 이렇게 NN이 벡터 내적으로 표현되는 것은 보통의 NN에서는 보기 어려운 경우인데 왜냐면 SG에서는 특별하게도 가운데 레이어가 하나 뿐이고, 게다가 가운데 레이어에는 Activation Function도 없기 때문에 가능한 것이기 때문이다.
그리고 sigmoid를 사용하기에 얼핏 Binary Classification과 같아 보이지만 같은 상황(w와 c)에 대해서 그때그때 다른 결과(0 또는 1)를 학습한다는 면에 Binary Classification과는 다르다. 이런 것을 Probalistic Binary Classification이라 한다. 마지막으로 위 과정에서 (w, c) 쌍을 뽑는 것에는 크게 의문이 없으나 (w, n) 쌍에서 사용하는 n은 어떻게 뽑을까? 역시 크게 복잡한 방법을 사용하는 것은 아니다. 간단히 보자면 문서 전체에서 자주 등장하는 단어는 자주 n으로 선정된다.
Word2Vec의 학습은 정말 이렇게 하는 것이다. 학습 시에도 추론 시에도 Softmax를 사용하지 않는다. 그래서 이 결론에 대해서만 알기 위해서는 Softmax 관련해서 알 필요는 없다. 다만 이 결론에 이르기까지 고생하고 고민하고 유도하고 하는 과정에서 Softmax가 등장할 뿐이다. Good과 Bad를 Target으로 학습한다는 면에서 GAN과 비슷한 느낌도 살짝 든다. 하지만 그것에 대한 내용은 이 글의 범위를 벗어나므로 여기서는 생략한다. 그리고 Iterative 하게, 점진적으로 학습을 해 나간다는 면에서 Minibatch를 이용한 Stochastic Gradient Descent와 뭔가 관련이 있어 보이기도 한다. 요즘 나오는 연구 결과들에 의하면 SGD는 GD만큼이 아니라 오히려 GD보다 좋다고 한다. 그러니까 Word2Vec에서도 이렇게 Iterative 하게 Stochastic 하게 돌려도 돌아갈 수 있을 것이다.
아무튼 그래서 Word2Vec은 위와 같이 매우 간단하다. Keras 코드로 보면 핵심 내용은 겨우 몇십 줄도 안 된다. 신경망 build와 train에 해당하는 코드 보다 오히려 data 준비하는 코드가 더 길다.

- Hierarchical softmax
    - 자료:
        - https://youtu.be/ioe1eeEWU0I
        - https://www.youtube.com/watch?v=vHNaRz0hdVw
    - Tree 구축 방법
        - 코드: https://talbaumel.github.io/blog/softmax/
    - 수식
    - 왜 잘 되는지 이유가 궁금함

# 2. Word Vectors 2 and Word Senses

LSA같은 방법은 global co-occurence를 쓸 수 있으나, 단어 간 유사도를 측정하기는 어렵다.
반면에 word2vec은 단어 간 유사도를 측정하는데 유리하지만, local context만 학습하기 때문에 corpus 전체의 통계정보는 반영하기 어렵다. GloVe는 이 둘의 장점만을 섞는 기법이다.

GloVe의 경우엔 word2vec과는 약간 다른 방법을 사용한다. word2vec이 neural network를 사용하여 학습한다면, GloVe는 global co-occurence matrix와 단어 i, j에 해당하는 embeddig matrix U*V간의 MSE loss를 최소화한다. 학습 후에는 Skip-gram과 같이 두 개의 matrix를 얻고, U를 보통 사용한다.

- $F(X_{ij})$
    - ![F of Glove](https://user-images.githubusercontent.com/47516855/96076940-836c1700-0ee9-11eb-8c9a-25247d7c4d6e.png)
