# Week 5: Linguistic Structure: *Dependency Parsing* (발표자: 유인혁)

**Contents**:
- Syntactic Structure
    - Consistency
    - Dependency
- Dependency Grammar and Treebanks
- Transition-based dependency parsing
- Neural dependency parsing

우선, 제목을 보면 **Linguistic Structure: Dependency parsing**라고 되어 있는데, dependency parsing이 뭔지 알아보자.

> **parsing(구문분석)**은 문장을 그것을 이루고 있는 구성 성분으로 분해하고 그들 사이의 위계 관계를 분석하여 문장의 구조를 결정하는 것을 말한다. Parsing에는 크게 두가지가 있는데 이는 constituency parsing과 dependency parsing이다.

> **dependency parsing(의존 구문 분석)**이란 *구문 분석*의 한 갈래이다. 의존 구문 분석 이론에서는 구문의 성분이나 성분을 연결하는 문법이 구문안에서 스스로 직접적인 역할을 한다기 보다는, 각 성분이 서로 의존 관계를 이루어 하나의 구문을 이룬다고 본다. 따라서 dependency parsing은 단어 간의 관계에 집중한다.

본 장의 목적은 사람이 쓰는 언어의 문장 구조를 이해하고, machine에게 사람이 쓰는 문장 구조를 이해시키기 위한 방법을 알아본다. 
본 장에서 알아볼 것들은 특정 언어가 아닌, 사람이 쓰는 모든 언어(e.g. 한국어, *프로그래밍언어*)에 적용된다.

# 1. Two views of linguistic structure

## 1.1 Consistuency(=phrase structure grammar=context-free grammar)

*Parse tree*란 올바른 문장에 대해 tree구조로 나타낸 것으로, NLP에서의 Parse tree는, compiler에서의 parse tree와 비슷하게, 문장의 구문구조/통사구조(syntactic structure)를 분석하는데 사용된다. 이러한 구조에는 크게 *constituency structure*와 *dependency strucutre*의 두 가지 구조가 있다.

constituency는 phrase structure grammar (구 구조 문법)를 사용하여 word를 nested *constituent*로 만든다. 여기서 phrase는 words의 조합이고, constituency strucutre의 기본 요소는 phrase이다. 이는 나중 챕터에서 다시 보도록 하자.

![consituency-based parse tree](https://upload.wikimedia.org/wikipedia/commons/5/54/Parse_tree_1.jpg)


## 1.2 Dependency structure

Dependency structure(의존 구문 분석)은 한 단어가 다른 단어의 종속적(수식 혹은 *논항*)인지를 보여준다. 이러한 단어들간의 비대칭적이고 binary한 관계를 dependency라고 부르고, **head** (or governor, superior, regent)에서 **dependent** (or modifier, inferior, subordinate)로의 화살표로 나타낸다. 일반적으로 이러한 dependency는 tree구조로 나타낸다. 또한 종종 문법적인 관계 (subject, prepositional object)를 써놓기도 한다.

> **논항**: 서술어가 의미를 성립시키기 위해 구조상 필요로 하게 되는 성분

![dependency-based parse tree](https://upload.wikimedia.org/wikipedia/commons/8/8c/Parse2.jpg)

그렇다면 이러한 sentence structure가 필요한 이유는 무엇일까? 우리는 언어를 올바르게 해석하기 위해 sentence structure를 이해하는 것이 필요하다. 또한, 사람은 복잡한 의미를 표현하기 위해 언어를 조합하여 더 큰 의미를 만들고, 이를 통해 의사소통을 하기 때문이다. 앞서 word2vec을 배웠지만, 이것만으로는 부족한 것이 있다. 따라서 우리는 무엇이 어디에 연결되는지 알 필요가 있다.

# 2. Dependency Grammar and Dependency Structure

> **dependency grammar**(의존 문법)은 뤼시앵 테니에르에 의해 발전된 통사 이론의 하나이다. 구 구조 문법(phrase structure grammar)과 구별되는 점은 구 노드(phrase node)가 없다는 것이다. 구조는 단어(핵)와 그 의존 요소 사이의 관계에 의해 결정된다. 의존 문법은 특정한 어순을 제한하지 않기 때문에 자유 어순 언어를 설명하는 데에 적합하다. 

Dependency syntax는 통사구조(syntactic structure)가 어휘 간의 관계를 구성한다고 가정한다. 이러한 단어들간의 비대칭적이고 binary한 관계를 dependency라고 부른다. 화살표를 그리는 방법은 다양한데, 여기서는 **수식 받음 -> 수식함**으로 나타내도록 한다. 다음은 문장 **Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas**의 dependency structure를 tree 형태로 표현한 예시이다.


![dependency](https://user-images.githubusercontent.com/47516855/97797273-1681ac80-1c5f-11eb-8603-af0b42e52898.png)


또한, 화살표는 보통 문법적인 관계와 함께 나타낸다.


![grammatical relations](https://user-images.githubusercontent.com/47516855/97797294-36b16b80-1c5f-11eb-8260-add0fd0c42ac.png)


일반적으로는 ROOT를 추구하여, 모든 단어가 정확히 하나의 단어에 dependent하도록 한다.

## 2.1. Universal Dependencies treebanks

Treebank는 문장 내의 dependency를 드러내는 annotated data이다. 해당 데이터셋은 다음 그림에서와 같이 단어간의 내의 dependency가 분석된 문장들로 이루어져있다.

![tree banks](https://pakalguksu.github.io/images/cs224n/lec5-2.png)

우선, treebank를 만드는 것은 문법을 만드는 것에 비해 굉장히 비효율적이지만, 그럼에도 불구하고 treebank는 다양한 장점이 있다:
- Reusability of the labor
    - Many parsers, POS taggers, etc. can by built on it
    - Valuable resource for linguistics
- Broad coverage, not just a few intuitions
- Frequencies and distributional information
- A way to evaluate systems


## 2.2 Dependency Conditioning Preferences

그럼 이를 어떻게 만들 것인가? 다음과 같은 것들을 고려해볼 수 있다.

![image](https://user-images.githubusercontent.com/47516855/97797743-dc1a0e80-1c62-11eb-8b85-9abd43f9ce9f.png)

1. Bilexical affinities
    - 실질적인 관계에 주목한다. 예를 들어 [discussion -> issues]는 타당하다
2. Dependency distance
    - 대부분은 가까이 있는 단어에 dependent하다
3. Intervening material
    - 동사나 구둣점을 뛰어 넘어서 이루어지지 않는다
4. Valency of heads
    - head에 따라서 dependents에 대한 패턴이 어느 정도 존재한다. 예를 들어, 관사 the는 dependent가 없는 반면, 동사는 많은 dependents 를 갖는다. noun 도 많은 dependents를 갖는데, 형용사 dependent는 (주로) 왼쪽에 위치하고, 전치사 dependent는 오른쪽에 위치한다


## 2.3 Dependency Parsing

Dependency parsing은 문장 S에 대해 syntactic dependency structure를 분석하는 것이다. 일반적으로 dependency parsing 문제는 S=w0w1...wn로부터 이의 dependency tree graph G와의 mapping을 만드는 것이다.

dependency parsing의 subproblem은 다음과 같다:
- Learning: annotated sentence set D와 dependency graph를 이용하여, parsing model M이 새로운 sentence를 parsing하게끔 하는 것
- Parsing: parsing model M과 sentence S에 대해, M에 따라 S의 최적의 dependency graph D를 도출하는 것

문장은 문장 내 단어가 어떤 단어의 dependent하는지에 따라서 parsing이 된다. 일반적으로는 다음과 같은 제약이 있다:
- 오직 한 단어만 ROOT의 dependent
- cycles은 허용하지 않음: A-> B, B -> A

이는 dependency를 tree로 만들 수 있게 한다. Final issue로는 화살표가 서로 cross 되는가 안 되는가 이다. 이를 *projectivity*라고 한다. CFG tree의 경우는 반드시 projectivity를 만족하지만, dependency같은 경우 일반적으로 displaced constituents를 설명하기 위해 non-projective를 허용한다.

dependency parsing의 subproblem은 다음과 같다:
- Learning: annotated sentence set D와 dependency graph를 이용하여, parsing model M이 새로운 sentence를 parsing하게끔 하는 것
- Parsing: parsing model M과 sentence S에 대해, M에 따라 S의 최적의 dependency graph D를 도출하는 것


# 3. Greedy transition-based parsing

dependency parsing을 위한 방법은 여러가지가 있는데, 본 수업에서는 **Transition-based parsing/deteministic dependency parsing**에 집중한다.

## 3.1 Basic transition-based dependency parser

![Basic transition-based dependency parser](https://pakalguksu.github.io/images/cs224n/lec5-4.png)

문장 안의 단어는 ‘stack’과 ‘buffer’에 위치한다. 처음엔 ROOT만 stack에, 나머지 단어들은 buffer에 위치한다. 각 단계마다 parser는 다음 행동 중 하나를 반복한다:
- Shift:  buffer의 top word를 (가장 왼쪽 단어) stack의 top position (가장 오른쪽)으로 이동 
- Left arc: stack의 top word를 화살표로 연결 (<-), 그리고 dependent 제거 
- Right arc: stack의 top word를 화살표로 연결 (->) 그리고 dependent 제거
- stack 에 [root]만 원소로 남거나, buffer가 비면 종료

이는 총 |R|*2 + 1개의 선택지가 된다.


다음은 **I ate fish**에 대한 예제이다.

![](https://pakalguksu.github.io/images/cs224n/lec5-5.png)

![](https://pakalguksu.github.io/images/cs224n/lec5-6.png)


이 경우는 상황에 따라 적절한 행동(transition)을 취했기 때문에 parsing에 성공했지만, parser가 항상 적절한 행동을 취할 것이라고 단정할 수는 없다.


## 3.2 MaltParser

그러나 우리에겐 *다음 행동을 어떻게 취할까?*에 대한 설명이 필요하다.

MaltParser는 위와 같은 parser에 다음 행동을 예측하기 위해 머신러닝 기술을 도입한 경우이다. 각 단계에서 어떤 행동을 취할지 선택할 때, 머신러닝 모델을 이용해 가장 가능성 높은 행동을 취하는 것이다.

MaltParser는 ‘stack의 top word와 POS’, ‘first in buffer word와 POS’등의 feature를 보고 다음 행동을 선택한다. 비록 MaltParser는 성능에 있어 dependency parsing의 SOTA에 못 미치지만 linear time 안에 매우 빠른 parsing을 할 수 있으며, 꽤 괜찮은 성능을 보인다.


# 4. Neural dependency parsing

Neural Dependency Parser는 현재 stack, buffer, dependency tree 상태에서 별도의 feature computation 없이 바로 다음 행동을 결정하는 parser이다.

기존의 parser들은 현재 상태를 파악하기 위해 stack과 buffer, dependency tree로부터 유용한 feature를 생성했다. 그런데, 이 feature는 굉장히 높은 dimensionality를 가지며, 매우 sparse한 성질을 가지고 있다. 또, 실제 문법적 규칙을 반영하는 feature인 indicator feature의 경우 사람이 직접 만들어줘야하며 엄청나게 많은 규칙이 필요하다. 이와 같은 문제때문에, 기존 parser들은 feature computation 과정에서 parsing의 95%가 넘는 시간을 소모했다.

Neural dependency parser는 neural network를 도입해 이 feature computation의 비효율성을 해소한다. 구체적으로 소개된 neural dependency parser의 동작 과정은 다음과 같다.
- 각 단어를 d차원 dense vector로 표현 (i.e. word2vec)
- 각 단어의 POS tag와 dependency label도 d차원 vector로 표현
- stack, buffer에서 단와 POS tag, dependency label을 추출하고 vector embedding으로 나타낸 후 concatenate한 결과를 neural net에 feed

![](https://pakalguksu.github.io/images/cs224n/lec5-8.png)


## 4.1 Evaluation

![image](https://user-images.githubusercontent.com/47516855/97798285-1043fe00-1c68-11eb-9b41-3368fab7a644.png)


- unlabeled attachment score (UAS) : arc(dependency)만 일치하는지 확인한다. label은 별도로 확인하지 않는다.
- labeled attachment score (LAS) : arc와 label 모두 일치하는지 확인한다.




- context free grammar: 
- 문맥 의존 문법(Context-sensitive grammar, CSG), 문맥 민감 문법은 형식 문법(formal grammar)의 한 종류로, 생성규칙에서 시작 부분과 끝부분을 나타내는 것을 포함하는 부분이다.
- 형식 문법(形式文法, formal grammar)은 형식 언어를 정의하는 방법으로, 유한개의 규칙을 통해 어떤 문자열이 특정 언어에 포함되는지를 판단하거나, 그 문법으로부터 어떤 문자열을 생성해 낼지를 정한다. 


- 