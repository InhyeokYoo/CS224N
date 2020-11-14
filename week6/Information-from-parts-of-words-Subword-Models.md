# Lecture 12: Information from parts of words: Subword Models (발표자: 유인혁)

**Lecture Plan**

본 시간에는 neural network관점에서 봤을 때 새로운 것은 없고, 매우 쉬울 것이다. 이 강의를 처음 계획했던 2014-2015년 경에는 NLP에 대한 모든 딥 러닝 모델이 단어 단위로 동작했다고 한다. 따라서 word vector로 출발하여 RNN같은 것들에 집어넣는게 매우 당연했다. 그러나 최근 3년 간 엄청나게 많고 새로운 것들이 튀어나왔고, 이 중 몇 몇은 매우 영향력 있는 모델이 되었다. 이 중에는 language modeling을 만드는 방법이 있는데, 단순히 단어를 기반으로 language modeling을 하는 것이 아니라 단어의 조각 혹은 character를 통해 modeling하는 것이다. 그리고 기존에 살펴보았던 RNN, CNN 등에 올리게 된다. 

다음은 이번 강의에 살펴볼 목록이다.

1. A tiny bit of linguistics
2. Purely character-level models
3. Subword-models: Byte Pair Encoding and friends
4. Hybrid character and word level models
5. fastText

# 1. A tiny bit of linguistics

다음은 언어학의 구조이다.

![Major levels of linguistic structure](https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Major_levels_of_linguistic_structure.svg/600px-Major_levels_of_linguistic_structure.svg.png)

## Human language sounds: Phonetics and phonology

우선 언어의 lower level unit에서 언어의 구조에 대해 배우는 것 부터 시작해보자.

언어학에서 좀 덜 중요한 것 (bottom of the totem pole)부터 보자면 *phonetic (음성학)* <sup>1</sup>이 있다. 이는 사람 말소리의 생리학과 소리를 이해하는 것으로, 음성의 물리적/생리적 측면을 연구하는 것이다.

> *phonetic*: 말소리의 실체에 물리적으로 접근하여 기술하고 분석하는 분야로, 물리적인 말소리의 생성과 음향 및 인지에 초점을 맞춘다. 음성학은 말소리를 만들기 위해서 움직이는 기관에 대한 관찰과 그것에 토대를 둔 말소리의 분류 및 만들어진 말소리의 음파, 그 음파가 귀로 들어가 뇌로 전달되는 과정 등을 다루는 분야로 인간이 말을 할 때 이용하는 소리를 물리적으로 분석한다. [출처: Ratsgo blog](https://ratsgo.github.io/speechbook/docs/phonetics)

그러나 이 윗단인 *phonology (음운학)*에서 실제로 언어를 이해하기 위한 표준은, 언어는 상대적으로 작은, 서로 구별되는 unit의 집합 만을 사용한다는 것이다. 이러한 집합은 phoneme (음소)라 부르고 categorical하다.

> *phonology*: 음성학과는 달리 말소리의 물리적 실체를 직접 다루기보단, 언어 사용자의 머릿속에 있는 말소리에 대한 지식을 체계적으로 기술하고 설명하는 분야이다. 음운론에서는 개별 언어들에서 사용되는 말소리의 특질과 그 언어 사용자가 가지고 있는 말소리에 대한 지식을 다룬다. [출처: Ratsgo blog](https://ratsgo.github.io/speechbook/docs/phonetics)

이렇게 하는 이유는 우리의 입은 continous space이고, 따라서 무한히 많은 종류의 소리를 낼 수 있다. 그러나 실제로 언어는 이렇게 무한한 소리를 만들어내기보단, 소리의 차이를 구분한다.

## Morphology: Parts of words

전통적인 언어학에서 소리는 아무런 의미가 없다. 따라서 언어학의 다음 단계인 morhology (형태론)을 사용하여 단어의 부분 (parts of words)을 구분하게 된다. 이는 의미의 최소 단위를 찾는다.

단어는 복잡하고, 의미를 갖는 작은 요소들로 이루어져있다. 예를 들어 *unfortunately*같은 경우는 다음과 같이 나눌 수 있다.
- un: 어떤 반대되는 개념
- fortun(e): 운
- ate: 형용사를 만듬
- ly: 부사를 만듬

이러한 단어의 작은 조각들은 의미를 갖는 최소한의 단위이다.

![image](https://user-images.githubusercontent.com/47516855/99148830-54ee8100-26cd-11eb-86d3-b0fbe0782402.png)


그러나 이러한 연구에는 딥러닝이 거의 활용되지 않는다. Manning 교수와 그의 제자들이 2013년에 트리 구조(recursive neural network)를 이용해 형태소 분석을 하려는 시도만이 있었을 뿐이다 (Luong, Socher, & Manning 2013). 그 이유로는 이 작업이 어려운 작업일 뿐만 아니라 character n-gram을 사용하면 비슷한 결과를 낼 수 있기 때문이다. 위와 같은 결과에 대해 character tri-gram을 통해 (<sos>, u, n), (u, n, f), ..., (l, y, <eos>)를 distributed하게 얻고, 이를 통해 모델이 단어에서 중요한 의미를 갖는 요소들만 뽑아내게끔 만들어 낼 수 있고, 적절하게 괜찮은 결과를 낼 수 있다. 물론 우리는 CNN을 배웠으니까, CNN을 쓰면 비슷한 결과를 뽑을 수도 있을 것이다.

그러면 형태소의 유용함은 그대로 유지하면서 이를 더 쉽게 뽑아내는 방법은 뭐가 있을까?

우선 단어를 쓰지 않고 모델을 만드는 경우를 생각해보자. 이 경우 단어를 character로 볼 수 있을 것이다. 이 경우 알아둬야 할 점은 언어마다 다양한 형태가 있다는 것이다.
- No word segmentation: 중국어, 일본어
- Words (mainly) segmented: 영어, 한국어

대부분의 언어는 띄어쓰기로 단어가 구분되는 형태이지만, 이에도 다양한 형태가 있다. 많은 언어들이 pronoun(대명사)와 전치사, 언어들이 결합한 형태를 갖고 있고, 이들은 때때로는 붙여서 쓰지만, 또 어떨때는 분리해서 쓰기도 한다.

![image](https://user-images.githubusercontent.com/47516855/99147394-025c9700-26c4-11eb-8bc2-ba7b064dc44b.png)

프랑스어의 경우 *clitic(전어)*, 대명사, *agreement(일치)*가 분리되어 있지만, 하나의 단어처럼 쓰인다. 반면, 아랍어에서는 함께 붙어서 하나의 단어로 쓰이지만, 실제로는 4개의 단어가 되야 한다.

또 다른 예시로는 복합명사(compound noun)가 있다. 영어에서는 복합명사를 띄어쓰기로 나누는게 기본이지만, whiteboard, behave, highschool처럼 붙여쓰는 경우도 있다. 그러나 독일어는 이 모두를 붙여서 쓴다.

이처럼 띄어쓰기만 사용하고 그 외에는 별 다른 조치를 취하지 않을 경우 다른 결과를 얻게 될 것이다.

> Clitic(전어): A morpheme that has syntactic characteristics of a word, but depends phonologically on another word or phrase.

> Agreement(일치): 어떤 단어가 문장 안의 다른 단어와 맺는 관계 때문에 그 형태가 변하는 현상. 굴절의 일종이며, 일반적으로 문장 내의 여러 단어나 성분들이 어떤 문법범주(성이나 인칭 따위)에 대해 같은 값을 갖도록 하는 과정이다. 예를 들어 표준 영어에서는 ‘I am’이나 ‘he is’라고 말할 수 있지만, ‘I is’나 ‘he am’이라고 말할 수는 없다. 영어에서는 동사와 그 주어가 인칭에서 일치해야 하기 때문이다. 영어 대명사 ‘I’, ‘he’는 각각 1인칭, 3인칭이고, 동사 형태 ‘am’, ‘is’도 각각 1인칭, 3인칭이다. 따라서 주어와 같은 인칭을 가진 형태가 선택된다.
 
## Model below the word level

![image](https://user-images.githubusercontent.com/47516855/99147854-85cbb780-26c7-11eb-84d5-0693ea269f2a.png)


이와같이 실제로 단어를 다루다보면 다양한 문제를 만나게된다.

word level의 문제점 중 하나는 큰 단어 사전을 다뤄야 한다는 점이다. 영어의 경우에는 다른 언어보다 문제가 더 많은데, 터키어로서는 한 단어로 표현할 수 있는 것을 영어로는 여러 글자에 걸쳐서 표현해야 한다.

번역을 할 때는 사람 이름처럼 고유명사도 신경을 써줘야 한다. 출발어의 발음구조와 유사한 발음을 도착어에서 찾아줘야 하고, 이를 위해서는 기본적으로 letter level에서 진행하는게 더 편할 것이다.

word level보다 더 작은 수준에서 모델링하는 다른 이유로는 slang과 abbreviation같이 social media로 인한 영향이 있다. 예시와 같이 단어를 늘려쓰기도 하고,  
이렇게 될 경우 word level로 진행할 경우 아주 큰 문제가 생길 것이고, 사람이 하는 것과도 굉장히 차이가 난다.

# 2. Purely character-level models

## Character-Level Models

이러한 문제로 character-level model이 등장하였다. 이에는 크게 두가지 방법이 있다.

**1. Character embedding을 통해 word embedding 구성하기**

character의 조합을 통해 unknown word에 대해 embedidng을 생성하고, 비슷한 character의 조합은 비슷한 embedding을 갖게 된다. 이로 인해 OOV문제가 없어진다.

**2. 언어를 chracter의 sequence로 생각하기**

중국어와 같은 connected language에 효과적이다. 

이 둘 모두 큰 성공을 거뒀다. Manning 교수는 처음에 이런 모델이 제안되었을 때 안될 것이라고 예상했다고 한다. word에는 각자의 의미가 있기 때문에 word2vec같은게 가능했고, 실제로 이로 인해 이들의 의미를 파악하는게 가능했다. 그러나 단어 하나하나를 본다는 것 자체가 의심스러웠다고 한다. 하지만 이는 empirical하게 증명됐다. 여기서 근본적으로 깨달아야 할 점은 글자 자체가 의미가 없긴 하지만, RNN, CNN같은 모델은 엄청나게 많은 파라미터가 있고, 이 파라미터로 인해 


# 3. Subword-models: Byte Pair Encoding and friends
# 4. Hybrid character and word level models
# 5. fastText