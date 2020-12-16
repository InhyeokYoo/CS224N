# Lecture 18: Tree Recursive Neural Networks, Constituency Parsing, and Sentiment (발표자: 유인혁)

**Materials**
- Youtube Lecture [바로가기](https://www.youtube.com/watch?v=6Z4A3RSf-HY&feature=youtu.be)
- Note: [바로가기](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes09-RecursiveNN_constituencyparsing.pdf)
- Slide: [바로가기](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture18-TreeRNNs.pdf)
- Suggested readings:
    - [Parsing with Compositional Vector Grammars](https://www.aclweb.org/anthology/P13-1045/)
    - [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/pdf/1805.01052.pdf)

**Lecture Plan**

1. Motivation: Compositionality and Recursion (10 mins)
2. Structure prediction with simple Tree RNN: Parsing (20 mins)
3. Backpropagation through Structure (5 mins)
4. More complex TreeRNNunits (35 mins)
5. Other uses of tree-recursive neural nets (5 mins)
6. Institute for Human-Centered Artificial Intelligence (5 mins)

이 시간에 배울 recursive tree는 2010년 경 스탠포드에서 NLP 강의를 시작했을 때 주로 작업했던 것 중 하나로, 주로 neural network를 활용하여 recursive tree를 만드는 것을 목표로 하였다. 이에는 언어 구조와 연관 깊은 개념들이 있으나, 실제로는 scale하기 어렵고 다른 개념 (LSTM, Transformer)보다 그렇게 중요하지는 않다. 이 때문에 강의의 막 바지로 순위가 밀리게 되었다.

**Last minute project tips**

다음은 CS224N의 과제를 위한 팁인데, 생각보다 괜찮은 내용이 많아서 가져왔다.

- 아무것도 되지 않고 모든게 너무 느리면 -> 패닉에 빠짐

이를 해결하기 위해, 

- 모델은 단순하게 만듬 -> 기본으로 돌아가자 (BOW + NNet)
- 아주 작은 모델과 데이터를 통해 디버깅을 할 것
- 버그가 없고 모델이 괜찮아보이면: size 키우자
- 항상 모델이 overfitting하는지 확인하고 조심할 것
    - 좋은 모델의 가장 기본적인 요건임
- train/dev error를 그려볼 것
    - 좋은 deep learning researcher라면 시간을 낭비하지 않음. 이를 통해 모델을 다시 학습할 정할 수 있음
- 문제가 없다면, L2 regularize와 Dropout을 할 것
    - 모델을 더 좋게 만들어야 함
- 시간이 남는다면 hyperparameter search를 함
- 조교한테 물어보러 올 것

# Consitituency Parsing

이는 슬라이드에는 없고 강의 note에만 있는 내용이라 옮겨서 적어본다. NLU는 큰 text units으로부터 의미를 뽑아내는 작업이 필요하다. 이는 더 작은 부품을 이해하는 것에서 출발한다. 문장의 syntatic strucutre를 분석하는 방법은 두 가지가 있는데, 이는 앞서 배웠던 **dependency parsing** (lecture4)과 **constituency parsing**이다.

dependency parsing은 단어와 이의 dependent 간의 비대칭적 binary relation(화살표)을 만들어서 단어가 어떤 단어에 종속하는지를 파악할 수 있다.

constituency parsing (=phrase structure parsing,  context free grammar)은 단어를 nested constituent로 만든다. 이는 piece of text (e.g. one sentence)를 sub-phrase로 나누는 작업이다. 이의 목표는 text에서 정보를 뽑아낼 때 유용한 contituent를 파악하는 것이다. parsing을 통해 constituent를 파악하면, 문법적으로 올바르면서 비슷한 문장을 생성해낼 수 있다.

## Constituent

syntactic analysis (also referred to as syntax analysis or parsing)에서 constituent는 single word/phrase가 될 수 있으며, hierarchical structure 내에서 하나의 unit으로 간주된다. Phrase란 둘 이상의 단어의 sequence로, head lexical item 주위에서 구축되며 하나의 unit으로 동작한다. phrase가 되려면 words의 그룹이 함께 문장 내에서 특정한 역할을 해야한다. 즉, 함께 이동, 교체 되야하고, 그럼에도 그 문장은 여전히 문법적으로 옳고, 뜻이 변해선 안된다.

예를 들어, 다음과 같은 noun phrase를 포함하는 문장이 있다고 해보자: *wonderful CS224N*
- *I want to be enrolled in the wonderful CS224N!*
여기서 phrase를 한 번 옮겨보자
- *The wonderful CS224N I want to be enrolled in!*
여전히 같은 뜻일 내포하고 있음을 알 수 있다. 또한 phrase는 대체할 수 있다고 말했다. 가령 *great CS course in Stanford about NLP and Deep Learning*로 대체한다고 해보자.
- *I want to be enrolled in the great CS course in Stanford about NLP and Deep Learning!*
여전히 동일한 뜻을 갖고 있고, 문법적으로도 문제가 없는 것을 볼 수 있다.

constituent parsing에서 기본적인 cluase structure는 binary division으로 clause를 subject (noun phrase NP)와 predicate (verb phrase VP)로 나누는 것을 의미한다. Binary division의 결과는 one-to-one/more mapping이 된다. 문장 내 각 element에 대해, tree structure에는 하나 이상의 node가 존재한다.
- S -> NP, VP
실제로 parsing process는 특정 유사한 규칙을 보여준다. 문장 기호 S로 시작하여 phrase structure 규칙을 연속적으로 적용하고, 추상 기호를 실제 단어로 대체하여 규칙을 추론한다. 이와 같이 추출 된 규칙을 기반으로 유사한 문장을 생성 할 수 있다. 규칙이 정확하다면 이 방법으로 생성 된 모든 문장은 문법적으로 옳아야 한다. 그러나 이런 문장이 문법적으로 옳다고 해도, semnatically nosensical할 수도 있다.
- **Colorless green ideas sleep furiously**
이는 노엄 촘스키가 저서 Syntactic Structures에서 고안한 문장이다. 아무리 문법적으로 옳더라도, 문맥적으로 말이 되지 않으면 문장이 아님을 말하기 위해 본 문장을 고안했다.

## Constituecy Parse Tree

흥미롭게도 constituent는 다른 constituent내에 중첩한다. 따라서 이를 잘 표현하는 것은 tree이다. 일반적으로 우리는 parsing 과정을 표현하기 위해 consistency parse tree를 사용한다.


# Motivation: Compositionality and Recursion

## The spectrum of language in CS

Tree recursive network의 motivation 관점에서 언어 및 언어 이론을 한 번 살펴보자. 다음 사진은 카네기 멜론 대학교의 설치 미술 작품인데, bag of words를 표현한 작품이다. 공중에 있는 주머니에 단어들이 들어있고, 밑에는 stop words가 떨어져있다. 

![image](https://user-images.githubusercontent.com/47516855/101978588-62c9fe80-3c99-11eb-96ad-e8462945f4f9.png)

NLP 모델에서 한 가지 흥미로운 사실은 BOW모델은 꽤나 많은 일을 할 수 있다는 점이다. 매닝 교수는 딥러닝 시대에도 더 그렇다고 말하고 있다. 우리는 그냥 word vector를 얻은 다음에 얘네들을 평균 혹은 max-pool하고 그 다음 sentence/document vector를 얻어서 classifier를 만드는게 꽤나 괜찮다고 한다.  

그러나 이는 언어학의 주요 관점과는 정반대이다. 언어학은 주로 매우 복잡한 형식주의(formalism)을 통해 발화(utterance) 구조를 파악하는데 중점을 둔다. 오른쪽 그림은 노엄 촘스키의 syntatic tree인데, 이렇듯 매우 복잡한 data 구조/명료한 구조가 언어학을 묘사하는데 사용되었다. 지금 보는 것처럼 BOW와 언어학에는 큰 gap이 있다. 이 중간 어딘가에 있는, 우리에게 도움이 되는 구조를 찾아야 한다.

## Semantic interpretation of language – Not just word vectors

언어를 의미적으로 해석하기 위해서는 word vector만 사용할 것이 아니라 더 큰 phrase의 의미를 알아야 한다. 다음과 같은 예제를 보자

- The *snowboarder* is leaping over a mogul
- *A person on a snowboard* jumps into the air

여기서 *snowboarder*와 *A person on a snowboard*는 기본적으로 같은 뜻이다. 따라서 이러한 **chunks**를 파악해야 한다. 이러한 chunk를 우리는 constituent/phrase라고 한다. 이들은 의미를 갖고 있고, 우리는 이를 비교하고 싶다.

앞서 chunks of language를 얻는 tool을 배운적이 있는데, 바로 CNN을 이용하는 것이다. 그러나 이에는 본질적으로 다른 것이 있는데, 이러한 의미를 갖는 chunks의 길이는 가변적이라는 것이다. 위의 문장은 2의 길이를 갖는 반면, 아래는 5의 길이를 갖는다.

따라서 이를 가능케하려면 이러한 constituent chunk를 neural network에서 표현해야 한다. 이는 tree structured neural network의 기본 아이디어가 된다.

이를 위해 각 개별요소의 뜻을 파악하고 이를 모아 더 큰 뜻으로 만들 수 있다. (semantic composition of smaller elelments). 이러한 **compositionality**는 언어 뿐만 아니고 다른 곳에서도 사용된다. 예를 들어 기계는 하위 부품들을 모아서 특정 기능을 수행한다. 만약 이 기계의 기능을 알고 싶다면, 하위 부품들의 기능을 먼저 살펴보고, 이를 조합하여 전체그림을 파악할 수 있을 것이다. 이는 또한 vision에서도 사용된다. 아래 그림 또한 특정 component로 분류할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/101979495-dc191f80-3ca0-11eb-861e-248f907d5ff4.png)

인간의 언어가 constituent와 같은 hierarchical 구조를 갖음에는 의심의 여지가 없다. 그렇지만 *언어가 recursive한가?*에 대해서는 갑론을박이 있을 것이다. 예를 들면 recursive하면 무한하게 반복할 수 있어야 하는데, 실제로 우리 언어가 그렇지는 않기 때문이다. 그럼에도 불구하고 이런 recursive 구조는 언어를 묘사하는데 매우 자연스럽다. 예를 들면 다음과 같은 문장은 아래와 같이 recursive하게 바뀔 수 있다.
- [The person standing next to [the man from [the company that purchased [the firm that you used to work at]]]]

Noun phrase안에는 또 다른 noun phrase가 있고, 그 안에는 또 다른 noun phrase가 있다. 이는 전형적인 recursive 구조이다.

![image](https://user-images.githubusercontent.com/47516855/102006504-f1578200-3d64-11eb-9034-19dee315e3e1.png)

# Structure prediction with simple Tree RNN: Parsing

## Building on Word Vector Space Models

![image](https://user-images.githubusercontent.com/47516855/102006643-ce799d80-3d65-11eb-843d-d7d6b0c4b32b.png)

constituent의 의미적 유사도를 찾아내기 위해서는 단순히 word 뿐만 아니라 이러한 constituent에도 의미를 부여해야 한다. 예를 들면 

- ”I went to the mall yesterday”
- ”We went shopping last week”, 
- ”They went to the store”

와 같은 phrase를 표현하는 것이라 생각해보자. 이 셋은 비슷한 뜻을 갖고 있으니 가까운 거리를 갖아야 한다. 이를 표현하는 방법은 word vector들의 composition을 통해 phrase vector를 output하는 것이다. 이 결과 output은 word vector space와 동일한 space에 있게 된다. 그렇다면 phrase를 어떻게 vector space로 mapping할까?

principle of compositionality는 문장의 뜻(vector)이 안의 단어들의 뜻과 이를 조합하는 방식에 따라 결정하는 것이다.

![image](https://user-images.githubusercontent.com/47516855/102007017-c111e280-3d68-11eb-9835-9bb217626001.png)

예를 들어 *the country of my birth*라는 phrase가 있을 때, 이는 위 그림과 같은 방식으로 단어의 뜻 (vector)를 조합하여 phrase 전체의 뜻을 도출하는 것이다. 이를 위해서는 (1) 올바른 문장 구조를 parsing해야하고, (2) 이러한 문장의 뜻을 잘 계산할 수 있어야 한다. 

![image](https://user-images.githubusercontent.com/47516855/102007700-a93d5d00-3d6e-11eb-8307-58961f0ce903.png)

이를 위해서 Constituency Sentence Parsing을 사용하여 문장의 구조를 분석하고, 이에 따른 의미를 계산하여 이러한 문장 의미의 vector space를 얻는다.

> Constituency parsing: 문장의 구조를 파악하여 parsing (e.g., NP + VP + etc...). (= Phrase strucutre grammar, consititeuncy grammar, Penn Tree Bank, CFG)

## Recursive Neural Networks for Structure Prediction

이를 위해 Recursive Neural Networks을 제안한다. 이는 Recurreunt Neural Network의 superset이라 보면 된다. 이 RNN은 nested hierarchy와 intrinsic recursive structure를 갖는 구조에 완벽한데, 왜냐면 문장은 다음과 같이 recursive한 성질이 있기 때문이다.
-  *A small crowd quietly enters the historical church*
    - *A small crowd* (Noun Phrase)
    - *quietly enters the historical church* (Verb Phrase)
        - *quietly enters*
        - *historical church*
        - etc

또한, RNN은 어떠한 길이의 문장도 처리할 수 있다는 장점이 있다.

RNN의 가장 밑단에는 word vector가 있고, 이를 재귀적으로 반복하여 더 큰 constituent의 의미를 계산한다. 아래의 그림처럼 *on the mat*의 의미를 알고 싶다면, 이들의 문장구조에 따라 단어들을 재귀적으로 neural network에 집어넣는다

![image](https://user-images.githubusercontent.com/47516855/102007932-65e3ee00-3d70-11eb-9124-758a729d117d.png)

구체적인 계산은 다음과 같다. p는 constituent의 representation으로, 하위 요소 c1, c2를 concat한 후 affine transformation과 non-linear function을 통해 계산된다. 또한, score도 나오게 되는데, 이는 이러한 연결이 그럴듯한지 아닌지를 나타낸다. 여기서 parameter W는 모든 node간에 share된다.

![image](https://user-images.githubusercontent.com/47516855/102009439-cbd57300-3d7a-11eb-9c54-058c2d859925.png)

또한, 이는 greedy하게 동작한다. 아래 그림처럼 가장 밑단에서 가장 그럴듯한 composition을 선정하고,

![image](https://user-images.githubusercontent.com/47516855/102009528-528a5000-3d7b-11eb-94ce-494ac3d4c86d.png)

이후 다시 greedy하게 가장 그럴듯한 조합을 recursive하게 만든다.

![image](https://user-images.githubusercontent.com/47516855/102009551-7baae080-3d7b-11eb-8b40-b31ae4f1f630.png)

그 결과 word vector space에 속하는 vector를 얻게 되고, 이것이 바로 sentence/phrase vector가 된다.

# Backpropagation through Structure

이 부분은 Manning교수가 기본적인 BPTT와 크게 다를 것이 없다고 넘어간다.

## Discussion: Simple TreeRNN

이런 simpe TreeRNN은 꽤나 잘 동작하지만 문장의 의미를 잘 잡지는 못 한다. 이는 근본적으로 네트워크를 너무 간단하게 설계했기 때문이다. 우리의 모델은 단순히 constituent 2개를 concat하고, 이에 weight matrix W를 곱한 형태인데, 이는 우선 constituent 사이의 interaction을 모델링한 것이 아니고, 모든 문법적 요소들에 대해 같은 weight를 사용한다는 한계가 있다. 즉, 하나의 인풋에 대해 얻은 optimal W가 다른 카테고리에서도 optimal할거라는 보장이 없다는 뜻이다. 

# More complex TreeRNNunits

## Version 2: Syntactically-Untied RNN (Socher, Bauer, Manning, Ng 2013)

앞서 봤던 TreeRNN과 유사하면서 context-free style constituency parsing를 잘하는 parser를 보자. 이는 parsing이 greedy해지는 것과는 거리가 있는 방법이다. 앞서 보았던 symbolic grammar가 sentence를 위한 트리 구조로는 적합하다. 그러나 문제는 문장의 의미를 계산하는 더 좋은 방법이 많이 있다.

따라서 Probabilistic Context Free Grammar를 이용하여 가능한 tree structure를 구상하고, K-best list를 뽑은 후에 이 중에서 제일 좋은 CFG가 무엇인지를 고르는 것이다. 이는 다이나믹 프로그래밍을 통해 효율적으로 계산할 수 있다. 이후 neural net을 통해 meaning representation을 계산할 수 있다. 

이를 위해 Syntactically-Untied RNN가 도입되었다. 문장 안의 node는 symbolic CFG에 대한 카테고리를 갖고 있다. 따라서 이 카테고리에 따라 weight matrix를 다르게 설정한다. 이를 통해 B와 C가 합쳐지는 parsing은 symbolic하다. 

![image](https://user-images.githubusercontent.com/47516855/102016389-069fd100-3da4-11eb-82e6-b0b9a78f89e3.png)

그러나 Beam search 내 candidate score에 대해 전부 matrix-vector product를 수행해줘야 해서 너무 느리다는 문제가 있다. 따라서 앞서 말했듯 PCFG를 사용하여 가장 그럴듯한 parser를 찾는다. 이러한 PCFG와 tree RNN이 합쳐진 구조를 **Compositional Vector Grammar**라 부른다.

## Related Work for parsing

Parsing과 관련된 연구는 다음과 같다.

- Resulting CVG Parser is related to previous work that extends PCFG parsers
- Klein and Manning (2003a) : manual feature engineering
- Petrovet al. (2006) : learning algorithm that splits and merges syntactic categories 
- Lexicalized parsers (Collins, 2003; Charniak, 2000): describe each category with a lexical item
- Hall and Klein (2012) combine several such annotation schemes in a factored parser. 
- CVGs extend these ideas from discrete representations to richer continuous ones

CVG는 그 당시에는 꽤나 괜찮은 constituency parser를 제공했다. 아래는 experimental result이다.

![image](https://user-images.githubusercontent.com/47516855/102017869-e7597180-3dac-11eb-8e1a-6f668e963400.png)

맨 위에 모델은 classic한 모델이고, 방금 배운 CVG는 최고까진 아니지만 이러한 system에서 벗어난 parser를 개발했다는데에 의미가 있다. 여기서 좀 더 흥미로운 점은 우리의 parser는 올바른 parse tree를 제공할 뿐만 아니라 의미를 계산할 수 있다는 것이다.

또한, 단순히 노드의 meaning representation 뿐만 확인하는게 아니라, 모델이 학습한 weight도 확인할 수 있다. category-specific한 weight matrix W는 children과 함께 의미를 만들어낸다. W는 diagonal matrix의 pair로, identity matrix이다. 크기는 2x1인데, child가 두개이기 때문이다. identity matrix를 이용하는 이유는 우선 이러한 두 개의 카테고리를 합치는데에 사전지식이 없기 때문에, 이 둘을 average하는데 최우선을 뒀다고 한다. identity matrix일 경우 두 벡터 모두 똑같이 참고할 수 있기 때문에 default semantic을 제공할 수 있다. 이를 통해 어떤 child가 더 중요한지를 배우게 된다. 예를 들어 *The cat and*의 경우, 접속사보단 NP가 더 중요하다는 것을 알 수 있다.

![image](https://user-images.githubusercontent.com/47516855/102247943-f7f02000-3f43-11eb-98b0-4086b441aad8.png)



## Version 3: Compositionality Through Recursive Matrix-Vector Spaces

그러나 이에도 한계는 있다. 마찬가지로 근본적인 원인은 구조 자체가 단순하기 때문이다. 여전히 우리는 W와 child를 곱하고 있는데, 이는 앞서 제기했던 word interaction을 해결하지 못한다. 예를 들어 *very*와 같은 수식하는 단어를 생각해보자. *very*와 함께 등장하는 단어를 interpolation하는 것은 *very* 그 자체의 뜻을 온전히 담고 있지 않다. 이는 사실 semantic linguistic theory에서 전형적으로 일어나는 일로, good은 뜻이 있는거고, very는 어떤 함수로 good을 받아 very good의 뜻일 내보낸다. 

이처럼 very는 부사로서 강조를 위해 쓰인다. 그렇다면 어떻게 하면 다른 vector를 강조시키는 vector를 얻을 수 있을까? 어떻게하면 다른 vector를 scale할 수 있을까? 우선 단어를 다른 단어에 곱하는 형태가 필요하다. 이러한 composition은 word matrix를 활용하는 방법과 일반적인 affine과 quadratic equation을 이용하는 방법이 있다.

## Matrix-vector RNNs [Socher, Huval, Bhat, Manning, & Ng, 2012]

우선 첫 번째 방법은 word matrix를 사용하는 것이다.
그렇다면 어떤 단어가 vector를 갖고, 어떤 단어가 matrix를 갖어야 하는가? 이는 정하기 꽤 까다로운 문제이다. good은 형용사로서 수식을 받기도 하지만 (부사) 수식을 하기도 한다 (명사). 따라서 모든 단어와 phrase가 vector와 matrix를 갖도록 하는 것이다.

각 단어마다 vector(a, b)와 matrix(A, B)가 둘 다 있고, 이를 compose할 때 바꾼 다음에 vector와 곱을 하여 (Ab, Ba) constituent의 의미를 계산한다. 의미는 아래의 그림 왼쪽처럼 계산된다. 그 후 concat한다.

matrix의 경우 두 개의 child matrix를 concat하고, $W_M \in \mathbb R^{n \times 2n} $의 weight를 통해 constituent의 matrix를 계산한다.

![image](https://user-images.githubusercontent.com/47516855/102113062-e93e3600-3e7b-11eb-9662-310a0612e376.png)

다음은 이에 대한 결과이다.

![image](https://user-images.githubusercontent.com/47516855/102115162-85693c80-3e7e-11eb-8b07-e9d9bd0b21ea.png)


## Classification of Semantic Relationships

- My [apartment] has a pretty large [kitchen]

위와 같은 문장이 있을 때 kitchen은 apartment의 일부이다. 따라서 이 둘의 관계는 component-whole relationship을 갖을 것이다.

아래와 같이 phrase가 "the [movie] showed [wars]"라면, 이 둘의 관계는 message와 topic이 될 것이다. 이와 같은 관계를 얻기 위해서는 이 둘의 관계를 모델링하고, 이를 neural net에 통과시켜 classification을 진행한다.

![image](https://user-images.githubusercontent.com/47516855/102115629-2821bb00-3e7f-11eb-990f-536d476d0ead.png)

아래는 이에 대한 결과인데, 기존의 RNN을 MV-RNN이 뛰어넘은 것을 통해 우리의 모델이 더 향상된 것을 확인할 수 있다 (74.8 -> 79.1). 

![image](https://user-images.githubusercontent.com/47516855/102116027-b72ed300-3e7f-11eb-9925-42856d79b651.png)

## Sentiment analysis

다음 그림은 Stanford Sentiment Treebank로, 11,855 sentences에 대해 215,154 phrases에 label을 붙인 것이다. 이러한 데이터 셋으로 학습시켰을 때 성능의 향상이 일어나는 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/47516855/102221229-3a563480-3f25-11eb-86b2-0c7fa267a0a8.png)

![image](https://user-images.githubusercontent.com/47516855/102221335-61146b00-3f25-11eb-83f7-f8f86a9982b1.png)

그러나 역시 이에도 문제가 있다. 우선 파라미터의 수가 너무 많다는 점이다 (matrix라서 제곱만큼 더 필요). 두번째로는 성능의 문제다. MV-RNN으로는 특정한 관계를 표현할 수 없었다.

**Negated Positives**
어떤 단어가 positive하고, 다른 단어가 이를 negative하게 바꾼다고 했을 때, 모델이 전체 문장의 sentiment를 바꿀 정도로 weight를 줄 수가 없다.

![image](https://user-images.githubusercontent.com/47516855/102116787-cf532200-3e80-11eb-8b1e-f6cd1ff313e4.png)

**Negated Negative**
예를 들어 not bad, not dull과 같은 것을 의미한다. 이는 not과 같은 negate와 부정문이 섞인 형태라서 negative에서 neutral로 바뀌어야 한다. 그러나 모델은 이를 잘 잡지 못 한다.

![image](https://user-images.githubusercontent.com/47516855/102117102-4092d500-3e81-11eb-8eb2-30f5015611f5.png)

**"X but Y conjunction**
X가 부정적고, Y가 긍정적이라면 전체 문장은 긍정적인 문장이다. MV-RNN은 이를 잘 잡지 못 한다.

![image](https://user-images.githubusercontent.com/47516855/102117308-80f25300-3e81-11eb-8e53-c28e163fb90c.png)


## Version 4: Recursive Neural Tensor Network

두 번째 해결 방법은 전형적인 affine에다가 quadratic equation을 사용하는 것이다. 기존의 concat vector $ \in \mathbb R^{2d}$에다가 affine function과 nonlinear를 통과하는 대신, quadratic과 nonlinear를 통과하게 된다.

이는 두 단어 사이의 attention을 가능하게 하는 구조이다. 따라서 두 단어 사이의 interaction이 가능하다. 그러나 하나의 scalar값을 내놓는다. 이를 해결하기 위해 중간 값을 matrix가 아닌 tensor로 바꾼다. 따라서 word vector를 결합하는 것에 따라 적절하게 tensor를 slice하고, 여러개의 scalar값을 이용하여 vector를 만든다. 따라서 tensor는 $\in \mathbb R^{2d \times 2d \times d}$ 차원이 된다.

standard layer는 앞서 보았던 TreeRNN과 같은 구조이다. 없어도 딱히 상관은 없지만, 그냥 넣었다고 한다.

![image](https://user-images.githubusercontent.com/47516855/102226858-41347580-3f2c-11eb-9506-2a6d13f290d7.png)

이를 통해 RNTN은 vector들간의 additive, mediated multiplicative interaction을 가능케한다. 이는 CNN보다는 특정 측면에서 좋은 결과를 나타내기도하고, parse tree가 없다는 장점이 있다.

sentence label에 대해서는 딱히 좋은 성능은 아니지만, Treebank 데이터에 대해서는 좋은 성능을 얻었다.

![image](https://user-images.githubusercontent.com/47516855/102227045-76d95e80-3f2c-11eb-9966-808733ce1b44.png)

앞서 언급했던 *X buy Y*같은 경우에도 잘 해결하는 것으로 나왔다.

![image](https://user-images.githubusercontent.com/47516855/102227270-b607af80-3f2c-11eb-8afd-18b4caa16547.png)

이는 negated negative/positive의 경우에도 마찬가지이다.

![image](https://user-images.githubusercontent.com/47516855/102227498-fc5d0e80-3f2c-11eb-9f91-e2a219bdf3f8.png)

이러한 아이디어는 흥미롭고, 언어학과 연관이 깊다. 그러나 여러 이유로 이러한 아이디어는 NLP에서 많이 활용되지 않았다. 이는 high dimensional vector를 활용하는 다양한 모델들이 좋은 성능을 내고 있고, GPU는 uniform computation 환경에서 더 좋은 성능을 낸다. LSTM이나 CNN 등인 determinant computation이기 때문에 GPU 친화적이다. 그러나 RNN은 sentence마다 다른 구조를 갖고 있기 때문에 batch화 할 수 있는 방법이 없다.

## Version 5: Improving Deep Learning Semantic Representations using a TreeLSTM

이 부분은 slide에는 있는데 강의에는 없다.

## Tree-to-tree Neural Networks for Program Translation [Chen, Liu, and Song NeurIPS2018]

다음은 프로그래밍 언어 사이의 번역에 이용한 사례이다. neural language에서는 그 모호함 때문에 올바른 parse tree를 선정하기 어렵지만, 프로그래밍 언어는 그렇지 않다.

이 연구에서는 encoder-decoder 구조를 사용했고, generation시에는 source tree로부터 attention을 사용했다.

![image](https://user-images.githubusercontent.com/47516855/102230822-a8ecbf80-3f30-11eb-8069-4bb893a0c918.png)
