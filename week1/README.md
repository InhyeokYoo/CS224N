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
