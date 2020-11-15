# Lecture 12: Information from parts of words: Subword Models (발표자: 유인혁)

**Lecture Plan**

본 시간에는 neural network관점에서 봤을 때 새로운 것은 없고, 매우 쉬울 것이다. 이 강의를 처음 계획했던 2014-2015년 경에는 NLP에 대한 모든 딥 러닝 모델이 단어 단위로 동작했다고 한다. 따라서 word vector로 출발하여 RNN같은 것들에 집어넣는게 매우 당연했다. 그러나 최근 3년 간 엄청나게 많고 새로운 것들이 튀어나왔고, 이 중 몇 몇은 매우 영향력 있는 모델이 되었다. 
이 중에는 language modeling을 만드는 방법이 있는데, 단순히 단어를 기반으로 language modeling을 하는 것이 아니라 단어의 조각 혹은 character를 통해 modeling하는 것이다. 그리고 기존에 살펴보았던 RNN, CNN 등에 올리게 된다. 

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

## words in writing systems

*사실 이 다음에는 맥락 상 character-level model이 나와야 할 것 같은데, 그러기보단 word를 다루면서 생기는 문제에 대해 이야기하고 있다. 따라서 원문을 밑에 함께 첨부하고 진행하겠다.*

> So now we might be interested in building models that aren't over words. So we are going to have a word written as characters and do something with it such as build character n-grams.

한 가지 알아둬야 할 점은 언어마다 다양한 형태가 있다는 것이다.
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

word level의 문제점 중 하나는 큰 단어 사전을 다뤄야 한다는 점이다. 영어의 경우에는 다른 언어보다 문제가 더 많은데, 터키어로서는 한 단어로 표현할 수 있는 것을 영어로는 여러 글자에 걸쳐서 표현해야 한다.

번역을 할 때는 사람 이름처럼 고유명사도 신경을 써줘야 한다. 출발어의 발음구조와 유사한 발음을 도착어에서 찾아줘야 하고, 이를 위해서는 기본적으로 letter level에서 진행하는게 더 편할 것이다.

word level보다 더 작은 수준에서 모델링하는 다른 이유로는 slang과 abbreviation같이 social media로 인한 영향이 있다. 예시와 같이 단어를 늘려쓰기도 하고,  
이렇게 될 경우 word level로 진행할 경우 아주 큰 문제가 생길 것이고, 사람이 하는 것과도 굉장히 차이가 난다.

## Character-Level Models

이러한 문제로 character-level model이 등장하였다. 이에는 크게 두가지 방법이 있다.

**1. Word embeddings can be composed from chracter embeddings**

기본적으로는 여전히 model을 단어 위에다 올리는 것이다. 그러나 어떠한 character sequence에도 word representation을 만드는 것이 목적이다. 따라서 character의 조합을 통해 unknown word에 대해 embedidng을 생성하고, 비슷한 character의 조합은 비슷한 embedding을 갖게 된다. 이로 인해 OOV문제가 없어진다.

**2. Connected langauge can be processed as characters**

앞선 단어들은 신경쓰지말고, 모든 language processing을 그냥 sequence of character로 진행하는 것이다.

그리고 이 둘 모두 큰 성공을 거뒀다. Manning 교수는 처음에 이런 모델이 제안되었을 때 안될 것이라고 예상했다고 한다. word에는 각자의 의미가 있기 때문에 word2vec같은게 가능했고, 실제로 이로 인해 이들의 의미를 파악하는게 가능했다. 그러나 단어 하나하나를 본다는 것 자체가 의심스러웠다고 한다. 하지만 이는 empirical하게 증명됐다. 여기서 근본적으로 깨달아야 할 점은 글자 자체가 의미가 없긴 하지만, RNN, CNN같은 모델은 엄청나게 많은 파라미터가 있고, 이 파라미터 덕분에 multi-letter의 그룹으로부터 의미를 representation할 수 있게 된다는 것이다.

## Below the word: Writing systems

문자 체계에서 character를 사용할 때 주의할 점을 생각해보자. 만약 우리가 언어학자라면 우리는 소리를 가장 최우선으로 생각하는 경향이 있을 것이다. 이러한 소리는 앞서 언급했던 phoneme이 될 것이다. 근본적으로는 딥러닝은 phoneme을 시도한 적이 전혀 없다. 전통적인 speech recognizer는 종종 phoneme을 사용하지만, 딥러닝에서는 많은 데이터를 필요로 하고, 이를 쉽게 얻는 방법은 쓰여진 형태로 얻는 것이기 때문이다. 이는 데이터 관점에서 납득할만하다. 그러나 한 가지 이상한 점은 character-level model을 구성할 때 언어의 문자 체계에 의존한다는 점이다. 그러나 인간의 문자 체계는 단순히 하나가 아니다.
- Phonemic (maybe digraphs): 스페인어 같은 문자 체계는 phonemic해서 어떤 letter들이 특정한 소리를 갖고 있고, 그 소리를 낼 수 있다
- Fossilized phonemic: 영어같은 경우는 철자 체계도 엉망이고 실제 소리랑도 다르지만 여전히 phonemic system이다
- Syllabic/moraic: 반면 다양한 unit을 사용하고, character를 통해 syllable을 표현하는 경우도 있다. 한국어가 대표적
- Ideographic (syllabic): 표의문자. 이 또한 syllabic system이지만, 실제 소리보다 글자의 수가 더 많고, 글자 하나하나가 뜻을 갖고 있다. 이 경우 morpheme이 한 character가 된다. 중국어가 대표적이다
- Combination of the above: 일본어의 경우 일부는 moraic하고, 일부는 표의 문자를 섞어 쓴다

따라서 어떤 문자냐에 따라서 어떻게 쓰는지가 달라질 것이다. character unit은 중국어에선 letter-trigram이 되서 세 개의 morepheme이 되지만, 영어 같은 경우는 T-H-O와 같은 chracter-trigram으로 아무런 의미가 없게 된다.  

# 2. Purely character-level models

우선 온전히 chracter-level만을 이용한 모델을 살펴보자. 이전 강의에서 sentence-level classification을 살펴봤었다. 이게 대표적인 purely character-level model이다. 매우 깊게 쌓은 CNN을 썼고 (Conneau, Schwenk, Lecun, Barrault. EACL 2017), 이러한 deep convolutional stack이 효과적임을 보였다.

## Purely chracter-level NMT models

이러한 character-level model은 NMT에서도 시도 되었는데, 처음엔 그렇게 만족스럽지 않았지만, character-level decoder의 성공을 시작으로 (Junyoung Chung, Kyunghyun Cho, Yoshua Bengio, 2016) 유망한 결과를 보이기 시작했다:
- Wang Ling, Isabel Trancoso, Chris Dyer, Alan Black, arXiv 2015
- ***Thang Luong, Christopher Manning, ACL 2016***
- Marta R. Costa-Jussà, José A. R. Fonollosa, ACL 2016
 
**Thang Luong, Christopher Manning, ACL 2016**

Thang Luong, Christopher Manning, ACL 2016는 영어와 체코어 WMT 2015에 대해 연구를 진행하였다. 체코어는 character level로 진행하기 좋은 언어인데, 아까 봤던 morphology로 이루어진 굉장히 긴 단어들이 많이 있기 때문이다. 따라서 이런 vocab 문제로 체코어에 대해 word-level로 진행한 모델은 좋은 성능을 내기 힘들다. 본 연구를 통해 word-level base line보다 pure character-level seq2seq NMT가 더 좋은 결과를 보였으나 너무 느리다는 단점이 있다 (3주).

![image](https://user-images.githubusercontent.com/47516855/99151193-ab16f080-26dc-11eb-9ee3-bd0bffe9c150.png)

결과를 보면 11-year-old라는 단어(파랑)가 잘 번역된 것을 볼 수 있다.

이러한 character-level model의 문제는 만약 LSTM 같은데에 넣는다면, 길이가 너무 길어지고, character에는 정보가 별로 없기 때문에 BPTT를 더 많이 해야한다.

**Fully Character-Level Neural Machine Translation without Explicit Segmentation**

이 다음해에 Jason Lee, Kyunghyun Cho, Thomas Hoffmann, 2017는 새로운 모델을 발표했는데, 출발어의 의미와 복잡함을 더 잘 이해하려는 시도를 했다.

![image](https://user-images.githubusercontent.com/47516855/99151298-7ce5e080-26dd-11eb-92ae-2b62cfa56f63.png)


인코더에선 character-level을 기반으로 4개의 convolution을 사용한 후에 highway network을 통과하였고, 그렇게 얻어진 word embedding에 대해 bidirectional GRU를 사용하여 source representation을 얻는다. 디코더는 앞서 봤던 모델과 동일하다.



**Stronger character results with depth in LSTM seq2seq model**

![image](https://user-images.githubusercontent.com/47516855/99151945-10b9ab80-26e2-11eb-8167-4201535a8e39.png)

다음은 Revisiting Character-Based Neural Machine Translation with Capacity and Compression. 2018.Cherry, Foster, Bapna, Firat, Macherey, Google AI는 word와 character-based model을 비교했다. English-French(좌), Czech-English(하)를 보면 큰 모델에서는 character model(blue)이 이기는 것을 볼 수 있다. 한 가지 재미있는 점은 morphological complexity에 따라 결과가 달라진다는 것이다. 체코어 같은 경우는 character-level model을 쓰는게 좋은 선택이지만, 프랑스어의 경우는 미비한 상승을 볼 수 있다.

이 모델은 bi-LSTM encoder와 uni-LSTM decoder를 사용하였다. 가장 간단한 모델(x축: 1x2+2)은 bi-LSTM encoder(1x2) + 2-layers LSTM decoder(+2)를 사용했다.

그러나 이에도 명확한 단점이 있는데 앞서 봤던 모델처럼 모델이 클수록 시간이 오래 걸린다는 것이다. Word-level은 레이어의 수가 많더라도 크게 문제가 되지 않는 것을 볼 수 있다.

# 3. Subword-models: Byte Pair Encoding and friends

이에는 두 가지 family가 있다.

**word piece model**:

기본적으로는 word-level model과 같은 구조를 갖지만 그러나 더 작은 unit인 "word pieces"를 사용하는 모델이다. 이를 word piece model이라 부른다. 이에 대한 가장 흔한 접근은 BPE라 부르는 모델이다.
- Sennrich, Haddow, Birch, ACL'16a
- Chung, Cho, Bengio, ACL'16

** Hybrid model:**

다른 모델은 Hybrid model로, main 모델은 word에 대해서 동작하지만, UNK에 대해 representation을 만들 때 lower-level or character를 사용하는 것이다.
- Costa-Jussa & Fonollosa, ACL'16
- Luong & Manning, ACL'16

## Byte Pair Encoding

원래는 압축을 위한 알고리즘으로 딥러닝과는 전혀 상관없지만, BPE를 쓰는게 어느 순간 표준으로 자리잡았고, pieces of words를 representation하는데 성능이 매우 좋다. 따라서 매우 큰 사전을 효과적으로 구축할 수 있다.

이 알고리즘은 가장 빈번하게 나오는 byte를 새로운 byte로 바꾸는 것이다. 그리고 이 작업을 계속해서 반복하여 새로운 byte를 만들어서 효과적인 압축을 꾀한다. 이러한 byte를 character n-gram으로 바꾸어 NLP에서 활용한다 (하지만 character로 변경하지 않고 그냥 byte를 사용하는 케이스도 있다고 한다).

그럼 구체적으로 어떻게 하는지 살펴보자.

![image](https://user-images.githubusercontent.com/47516855/99183917-d05e3a00-2782-11eb-9b57-1c52c189e6b6.png)

이 작업은 데이터에 있는 모든 character(Unicode)에 대한 unigram vocabulary부터 시작한다. 모든 단어를 character로 분해한다.

```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
l o w : 5
l o w e r : 2
n e w e s t : 6
w i d e s t : 3

# Vocabulary
l, o, w, e, r, n, w, s, t, i, d
```

그리고 몇 번 iteration할지를 정한다. 총 10번 iterate로 가정하자. 가장 많은 빈도수를 갖는 것은 es이다.

```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
l o w : 5
l o w e r : 2
n e w es t : 6
w i d es t : 3

# Vocabulary at iter 1: 9 times for es
l, o, w, e, r, n, w, s, t, i, d, es
```

그 다음 많은 것은 es와 t이다.

```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
l o w : 5
l o w e r : 2
n e w est : 6
w i d est : 3

# Vocabulary at iter 2: 9 times for est
l, o, w, e, r, n, w, s, t, i, d, es, est
```
그 다음은 lo가 7번으로 가장 많다.

```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
lo w : 5
lo w e r : 2
n e w est : 6
w i d est : 3

# Vocabulary at iter 3: 7 times for lo
l, o, w, e, r, n, w, s, t, i, d, es, est, lo
```

그 다음은 low가 7번.

```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
low : 5
low e r : 2
n e w est : 6
w i d est : 3

# Vocabulary at iter 3: 7 times for low
l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low
```

최종적으로는 다음과 같이 된다.
```python
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
low : 5,
low e r : 2,
newest : 6,
widest : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
```

만일 lowest라는 단어가 등장한다면, 기존 vocab에서는 OOV가 된다. 그러나 BPE를 이용하면 <low, est>로 합칠 수 있다.

아래는 원저자 Sennrich가 논문에서 공개한 코드이다.
<script src="https://gist.github.com/InhyeokYoo/f68846e3e3bfd9abb367daf83157daab.js"></script>

이러한 segmentation은 prior tokenizer (MT에서 일반적으로 쓰이는 Moses같은)를 통해 확인된 단어에 대해서만 진행한다.

## Wordpiece/Sentencepiece model

구글의 NMT에선 BPE의 변형을 이용하였다. BPE랑 정확히 같은 알고리즘을 사용하기보단 살짝 수정하여 그들의 LM에다가 넣는 방식을 취했다. 이들은 단순하게 빈도수를 세는 것보단 LM의 perplexity를 가장 많이 감소시키는 (i.e. maximizing LM log likelihood) pair를 greedy하게 선정하여 넣는 방식을 취했다. 이에는 두 가지 버전이 있는데, 첫 번째는 **wordpiece model**이고 v2는 **sentencepiece model**이다.

tokenization의 결과에 대해 sub-word model을 적용하게 되면, 모든 언어에 대해서 tokenizer가 필요하게 된다. 따라서 tokenizer를 쓰는 대신 whitespace를 special token인 _으로 바꾼 뒤, groupping 한다.

BERT의 경우 wordpiece model의 변형을 사용하였는데, 일반적인 단어일 경우 그대로 집어 넣고, 희귀한 단어의 경우는 wordpiece로부터 생성한다.
- hypatia = h ##yp ##ati ##a

또 다른 방법으로는 character를 써서 무한히 많은 vocabulary를 써서 표현하는 것이다. 대신 이러한 것을 더 큰 시스템의 일부로서 동작하게 만든다.

![image](https://user-images.githubusercontent.com/47516855/99185064-b88ab400-278a-11eb-96f6-002f4b9fe3a2.png)

위 그림은 2014년에 등장한 것으로 이러한 것들의 초기 시도 중 하나이다. Character에 대해 convolution을 사용하여 word embedding을 만들고, 이를 high level model에서 사용한다.

![image](https://user-images.githubusercontent.com/47516855/99185167-7b72f180-278b-11eb-92b5-67777ce8385d.png)

혹은 LSTM을 사용해서 word representation을 할 수도 있다. 이 경우엔 bi-LSTM은 word representation을 만들고, 이를 통해서 language modeling을 만들었다.

![image](https://user-images.githubusercontent.com/47516855/99185392-ea9d1580-278c-11eb-98f2-522e4de66eb9.png)


이보다 조금 더 복잡한 것으로는 ELMo에서 사용되었던 "Character-Aware Neural Language Model" (Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. 2015)이 있다. 여러 언어에 대해 효과적이고 robust한 LM이고 eventful, eventfully, uneventful과 같은 relation을 encoder하며 동시에 rare word문제에서 자유롭다. 또한, 적은 parameter를 통해 비교적 높은 표현을 얻었다. CNN의 자세한 사항은 아래와 같다.

![image](https://user-images.githubusercontent.com/47516855/99185502-9b0b1980-278d-11eb-8409-bd6e43cd4fcb.png)

그림처럼 단어를 character로 나누어 embedding하고, 이에 대해 다양한 filter로 convolution 연산을 실행한다. 이후 각 character에 대한 convolution 중 가장 큰 값을 pooling하여 word embedding을 한다. 이는 단어를 가장 잘 표현하는 char-n-gram을 뽑는 것과 동일한 과정이다.

이를 토치로 구현한 것은 [다음](https://github.com/InhyeokYoo/NLP/blob/master/papers/4.ELMo/char_cnn.py#L38)을 통해 확인할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/99185811-98112880-278f-11eb-9309-18c75ab59af1.png)

![image](https://user-images.githubusercontent.com/47516855/99185852-d4448900-278f-11eb-834e-21fa82f547cd.png)


이후 위 그림들과 같이 highway network와 LSTM을 통과한다. 

![image](https://user-images.githubusercontent.com/47516855/99185988-8d0ac800-2790-11eb-8560-cfdb5e5d57f3.png)

이 결과 앞서 Manning교수가 밝혔던 회의적인 입장과는 다르게 더 작은 parameter를 갖으면서 좋은 결과를 보였다. 위는 이에 대한 결과이다.

![image](https://user-images.githubusercontent.com/47516855/99186161-d0b20180-2791-11eb-8923-e558c56da5bb.png)

모델이 내놓는 representation을 살펴보면 흥미로운 것을 발견할 수 있는데, 만일 모델이 highway netowrk를 통과하지 않으면 character의 영향력이 짙게 남아있다. 따라서 character가 비슷한 애들끼리 similarity가 비슷해지게 된다. 반면, highway network를 통과하게 되면 semantic similarity를 잘 포착하게 된다.


![image](https://user-images.githubusercontent.com/47516855/99186302-bc223900-2792-11eb-917d-0619f2a3f29b.png)

또한, OOV단어의 sematic도 잘 포착하는 것을 볼 수 있다. 이를 visualize하면 prefix, suffix, 심지어 하이픈까지도 word의 한 요소로서 잘 활용하고 있는 것을 볼 수 있다.

본 논문에서는 LM을 위해 word embedding을 하는 것의 필요성에 의문을 제기했고, CNNs + highway network를 통해 풍부한 semantic/structural 정보를 추출할 수 있음을 보였다.

# 4. Hybrid character and word level models



# 5. fastText