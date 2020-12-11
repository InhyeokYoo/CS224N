# Lecture 13: Modeling contexts of use: Contextual Representations and Pretraining (발표자: 유인혁)

**Youtube Lecture**:
- [바로가기](https://www.youtube.com/watch?v=S-CspeZ8FHc&feature=youtu.be)

**Suggested readings**:

- [Smith, Noah A. Contextual Word Representations: A Contextual Introduction. (Published just in time for this lecture!)](https://arxiv.org/pdf/1902.06006.pdf)
- [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)

위 둘은 lecture에서 제안하고 있는 material이다.

**Lecture Plan**

1. Reflections on word representations (10 mins)
2. Pre-ELMo and ELMO (20 mins)
3. ULMfit and onward (10 mins)
4. Transformer architectures (20 mins)
5. BERT (20 mins)

# 1. Reflections on word representations

지금까지 우리는 단어에 대한 한 가지 표현만을 배워왔다. 앞서 배운 word vector로는 word2vec, GloVe, fastText같은 것들이 있다. 

|                                                       | POS WSJ (acc.) | NER CoNLL (F1) |
| ----------------------------------------------------- | -------------- | -------------- |
| SOTA*                                                 | 97.24          | 89.31          |
| Supervised NN                                         | **96.37**      | **81.47**      |
| Unsupervised pre-training followed by supervised NN** | **97.20**      | **88.87**      |
| + hand-crafted features***                            | 97.29          | 89.59          |

다음표는 2012년 ACL tutorial에서 Manning교수가 발표했던 내용으로, POS tagging과 NER에 대한 결과이다. 맨 윗줄은 2000년대에 traditional categorical feature based classifier를 통해 달성한 SOTA이다. 보다시피 NN을 쓰는게 그렇게 좋은 생각은 아니었다. 따라서 2000년대에는 CRF라던가 SVM같은게 더 유행했었다. 그러다가 떠오른 것이 바로 unsupervised pre-training word vector를 사용하는 것이었다. 무려 130,000 word-embedding이 Wikipedia/Reuters data를 통해 학습됐고, window size는 11, 차원은 100이다. 이는 학습하는데 무려 7주나 걸렸다고 한다. 이로 인한 NER 결과는 거의 feature-based classifier와 비슷한 성과를 달성하게 되었다.

![](https://user-images.githubusercontent.com/47516855/99881474-63a0df00-2c5d-11eb-82e5-ab1619f05a4d.png)

2014-2018년 경에는 이와는 다른 양상을 보이기 시작했다. 바로 random initialization을 쓰는 것이다. 다음 그림은 앞서 보았던 dependancy parser (Chen and Manning)에서 나온 것인데, 이는 상당히 작은 corpus에서 학습했음에도 불구하고, approximation이 꽤나 괜찮은 성능을 보였다. 

그러나 pre-trained word embedding을 사용했더니 1%정도 성능이 상승했다. Random initializing도 좋은 결과를 보였지만, unsupervised pre-training에서 사용할 경우 더 많은 데이터로, 더 많이 학습할 수 있기 때문에 좋은 선택이 된다. 

## Tips for unknown words with word vectors

여기서 잠시 삼천포로 빠져서 unknown word가 나왔을 때 다루는 방법을 보자. 가장 간단하고 일방적인 방법은 다음과 같다.

- Train
  - Vocab = {count(word) >= 5 } + {\<UNK>}
  - 나머지 단어는 모두 \<UNK>로 학습
- Runtime
  - Use \<UNK> when OOV word occur

그러나 이런 경우에는 서로다른 UNK 단어를 meaning/identity 둘 다 구분할 방법이 없게 되어 좋은 방법이 아니다. 그러면 어떤 방법을 쓸 수 있을까? 우리는 [이전 시간](../week6/Information-from-parts-of-words-Subword-Models.md)에 char-level을 배웠으니까 이를 쓸 수 있을 것이다. 

특별히 question answering같은 경우에는 다른 방법을 쓸 수도 있다. QA는 OOV word라도 이의 word identity를 맞추는게 중요하기 때문이다. 이는 (Dhingra, Liu, Salakhutdinov, Cohen, 2017)에서 따온 방법이다.

- Unsupervised pre-training task에서는 test time에서 UNK가 등장하는 경우 pre-trained model에 있는 단어를 쓴다
- 새롭게 등장하는 경우 그냥 random vector로 초기화해서 쓴 다음에 vocab에 추가한다
  - 이러면 각 단어가 unique identity를 갖게 되는 효과가 있다. 즉, question과 potential answer에서 같은 단어를 본다면, 이 둘이 완벽하게 match된다. UNK는 이러한 효과를 기대하기가 힘들다 

전자의 경우 확실히 도움이 되는 방법이고, 후자는 큰 도움이 되지는 않는다. 또, 시도할 수 있는 방법은 word classes로 collapsing하는 것이다. 이는 unknown number, capitalzed thing 등을 \<UNK-class>로 변환하는 것이다.

## Representations for a word

자 다시 돌아와서, 앞서 우리는 단어에 대한 한 가지 표현만을 배웠다고 했다. 물론 여태까지 봤듯이 잘 작동했다. 그러나 이에는 두 가지 치명적인 문제가 있다.

- word type에 대해서 같은 representation을 얻는다. 이는 context를 무시한다
- 우리는 한 가지 표현만을 얻었다. 그러나 단어는 semantics, syntatic behavior, register/connotations 등을 포함하여 다른 **aspect**를 갖는다

> Register: 언어학에서 사용역(register)은 계층이나 연령, 지역, 문체 등에 따라 달리 나타나는 언어변이형의 하나이다. 일반어에 대해 전문어나 유아어, 지역 방언과 계층 방언, 속어 등이 이에 속한다.

> Connotation: A connotation is a commonly understood cultural or emotional association that some word or phrase carries, in addition to its explicit or literal meaning, which is its denotation.

이러한 문제를 풀려면 어떻게 해야 할까? 우리는 LM파트에서 LSTM을 통한 LM을 배운적이 있다. 이 LSTM이 하는 일은 매번 단어를 입력으로 받아 다음 단어를 예측하는 것이다. 이는 LSTM이 각 time-step에서 context-specific한 word representation을 생성해내는 것으로 생각해볼 수 있다.

![image](https://user-images.githubusercontent.com/47516855/99884134-162d6d80-2c6f-11eb-9e9d-e74bb53a7c6a.png)

# 2. Pre-ELMo and ELMO

## Peters et al. (2017): TagLM - "Pre-ELMo"

Peters et al.은 [Semi-supervised sequence tagging with bidirectional language model](https://arxiv.org/pdf/1705.00108.pdf) 는 최근 context-sensitive word embedding의 원조격인 논문으로, TagLM이라고 불린다. 이 논문에서는 context vector를 얻되, 작은 task-specific data (e.g. NER)에 대해서 RNN을 학습하기를 원했다. 이를 위해 선택한 방법은 **semi-supervised learning**이다.

![image](https://user-images.githubusercontent.com/47516855/99896867-e36b8f80-2cd7-11eb-9612-85486f691ae3.png)

위 그림은 본 논문의 architecture이다. 자세한 설명은 다음과 같다.

1. Large unlabeled data를 통해 word2vec과 같은 word embedding과 language model을 생성
2. word에 대한 word embedding representation과 LM representation을 얻는다
3. 그리고 이 둘 모두를 이용해 sequence tagging 모델을 학습시킨다.

다음은 이를 더 자세히 묘사한 그림이다.

![](https://user-images.githubusercontent.com/47516855/99897332-e799ac00-2cdb-11eb-99ce-80c059a5d3fb.png)

- 대량의 unsupervised data를 통해 bi-LSTM LM을 학습시킨 다음 (파랑색 박스) forward/backward representation을 concatenation하여 LM representation을 얻는다.
- word embedding은 두 가지를 concat하는데,
  - 하나는 일반적인 word2vec 스타일의 representation이고 (초록색), 
  - 나머지 하나는 character-level model을 통해 word representation을 얻는 것이다 (주황색). 
  - 이 둘은 concat되어 bi-LSTM을 통과한다 (바로 위의 자주색 사다리꼴).
- 그 후 LM representation과 word embedding은 concat되어 (노란색) bi-LSTM을 통과하고, NER tagging을 수행하게 된다.

여기서 중요한 사실은 bi-LSTM으로부터 얻은 representation이 매우 유용하다는 것이다. 그냥 supervised model에 이를 집어넣기만 했는데도 불구하고 word에 대해 더 나은 feature (meaning, context)를 제공한다.

Language model은 Billion word benchmark셋에 있는 800M개의 단어로 학습하였다. Language model을 통해 관측한 것으로는 다음이 있다.

- Supervised data로 학습한 LM은 도움이 되지 않음
  - *원문을 통해 확인한 결과 supervised set 자체의 문제보다는 데이터 크기의 영향으로 보임*
- Uni-directional 보다 bi-directional이 더 좋음
- 더 큰 사이즈의 LM (ppl 30)이 작은 모델 (ppl 48) 보다 더 도움 됨

## Also in the air: McCann et al. 2017: CoVe

[Contextualized Word Vectors](https://arxiv.org/pdf/1708.00107.pdf) (CoVe)는 Manning 교수가 그냥 넘어갔기 때문에 간략하게 슬라이드만 번역하도록 하겠다.

이 모델은 다른 NLP task에 context를 제공하기 위해 trained sequence model을 이용한다. 이들은 LM을 학습시키기 위해 NMT를 사용했는데, 이는 machine translation이 의미를 보존하는 역할을 하니까, 이게 좋은 objective라고 생각했다.

Context provider로 seq2seq + attention의 2-layers bi-LSTM을 사용, 다양한 task에서 GloVe를 능가하였다. 그러나 본 강의의 다른 pre-trained model보다 낫지 않기 때문에 이 정도만 알아보도록 하겠다.

## Peters et al. (2018): ELMo: Embeddings from Language Models

다음은 그 유명한 ELMo이다. 원문은 [Deep contextualized word representations. NAACL 2018](https://arxiv.org/abs/1802.05365)이다. 이에 대한 자세한 설명은 발표자의 깃허브에서 볼 수 있다.

- [Deep contextualized word representations (ELMO) review](https://inhyeokyoo.github.io/project/nlp/elmo-review/)
- [PyTorch ELMo Implementation](https://github.com/InhyeokYoo/NLP/tree/master/papers/4.ELMo)

이전에 봤던 TagLM의 저자들은 그 다음 해 더욱 발전된 시스템 ELMo를 선보였다. ELMo는 TagLM과는 다르게 bi-LSTM에서는 word embedding은 쓰지 않고 CNN을 활용한 character-level embedding만 사용했다. 이는 좀 더 compact LM을 의도하여 사람들이 다른 task에서 더 쉽게 사용하게 하기 위함이다 (RNN에 비해 parameter 수가 감소). 자세한 사항은 아래와 같다.

- 퍼포먼스를 중시하되 LM 사이즈를 너무 크지 않도록 노력
- 2 bi-LSTM layers
- CNN을 사용하여 초기 word vector를 생성
  - 2048 n-gram filters, 2 highway layers, 512 dim projection
- LSTM에서 4096 dim layer를 이용한 후, 512 dim projection을 사용하여 next input으로 사용
- residual connection 이용
- input token and softmax layer의 weight가 tie (share) => parameter tie regularization?

ELMo는 biLM representation의 task-specific combination을 학습한다. 이는 이전의 LSTM의 top layer만 사용하던 방식보다 더 진보된 방식이다. 

![](https://user-images.githubusercontent.com/47516855/99899133-32222500-2cea-11eb-82c5-82b6001dad34.png)

이후에 이를 사용할 때는 다음과 같이 한다.

- supervised model의 목적에 맞게 weight를 freeze
- ELMo weights를 concat하여 task specific model에 집어넣음
  - 자세한 사항은 task에 따라 달라짐
    - TagLM처럼 중간에다가 집어넣는게 typical
    - QA/generation과 같이 ELMo representation을 다시 집어 넣는 경우도 가능

얻어진 bi-LSTM representation은 layer에 따라 서로 다른 uses/meanings를 갖는다. lower-layer에서는 low-level syntax를 갖는데 유용하고 (POS tagging, syntatic dependencies, NER), higher layer에서는 high-level semantic을 포착하는데 유리하다 (Sentiment, semantic role labeling, QA, SNLI).

# 3. ULMfit and on ward

[Howard and Ruder (2018) Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)

Universal Language Model Fine-tuning for Text Classification (ULMfit)은 ELMo랑 비슷하게 2018년 경에 나온 논문이다. ULMfit은 big language model을 학습하고 target task에 대해 transfer learning을 진행하는 것이다.  

![](https://user-images.githubusercontent.com/47516855/100495087-5919a480-318b-11eb-8cc1-57c15946886a.png)

이들은 big general domain corpus (unsupervised)에 대해 LM을 학습하고 (biLM), target task data에 대해 tuning을 진행한다. 그리고 target task에 대해 fine-tune을 진행한다. 그러나 한 가지 특이한 점이 있는데, 그냥 단순히 LM feature를 다른 네트워크에서 사용하는 것이 아니라, 네트워크 구조는 유지하되 맨 윗단에 다른 objective를 붙였다. 이는 transformer을 사용하는 연구에도 영향을 미치게 되었다. 이는 맨 위에 있는 softmax parameter를 고정시키고 (그림 c의 black box),  대신에 다른 prediction unit을 붙여 수행할 수 있다. 

![](https://user-images.githubusercontent.com/47516855/100496570-7608a480-3198-11eb-9a52-5f8616f0ead8.png)

> 이 부분에서 살을 좀 더 붙여보면, 대부분의 모델들은 embedding 시에 이러한 pre-trained model을 활용하긴 했지만 중간 레이어들은 여전히 randomly initializing을 사용했다. pre-trained 모델을 사용하는 것이 일종의 transfer learning이기는 해도, 부분적인 적용일 뿐 중간 레이어에 이를 연결하는 방법에 대해서는 연구된 바가 없었다. 출처: [전이 학습 기반 NLP (2): ULMFiT](https://brunch.co.kr/@learning/13)

ULMfit은 하나의 GPU로 학습할 수 있을만큼 크지 않다. 또한, 논문에선 모델의 성능을 높이기 위해 많은 디테일, 트릭, 학습 시 신경써야 할 점 등을 소개하고 있다.  아래는 이에 대한 정리이다.

- LM fine-tuning (b)
  - discriminative fine-tuning: 레이어마다 다른 type의 정보를 갖으므로 서로 다른 learning rate를 적용
  - slanted triangular learning rates: learning rate scheduler
- classificer fine-tuning (c) 
  - Concat-pooling: 마지막 hidden state 뿐만 아니라 모든 hidden state의 max pooling, mean pooling을 concatenation하여 시간에 대한 정보손실을 막음
  - gradual layer unfreezing: 한 번에 학습할 경우 catastrophic forgetting이 일어나게 되므로, least general knowledge를 갖는 last layer부터 점차적으로 unfreezing하며 fine-tuning을 함.

이러한 pre-training은 text classification에서 매우 효과적인 것으로 드러났다.

![image](https://user-images.githubusercontent.com/47516855/100499246-40b98200-31ab-11eb-825c-0fd755f1b8db.png)

다음 그림은 성능을 비교한 그래프인데, supervised data로부터 좋은 성능을 내기 위해서는 엄청나게 많은 데이터가 필요하게 된다 (파랑색). 만일 unsupervised pre-trained data를 활용하여 transfer learning을 하게 된다면, scratch부터 학습한 것보다 훨씬 더 적은 데이터로도 비슷한 성능을 낼 수 있다 (주황색). 또한, target domain extra fine-tuning 또한 매우 효과적인 것으로 드러났다 (초록색)

![](https://user-images.githubusercontent.com/47516855/100499258-66df2200-31ab-11eb-889b-065ab150f5fd.png)

이러한 pre-trained model의 효율성을 깨달은 연구자들은 점차 모델의 사이즈를 늘려나가기 시작했다.

![](https://user-images.githubusercontent.com/47516855/100505906-436aa600-31b0-11eb-94d0-4a67bb7b686f.png)

이 중 맨 왼쪽의 ULMfit을 제외한 나머지 모델은 transformer 기반의 모델들로, 효율적일 뿐만 아니라 훨씬 큰 사이즈로 scailing까지 가능한 구조이다. 이에 대한 이해를 돕기 위해 transformer를 살펴보자.

![](https://user-images.githubusercontent.com/47516855/100510647-b0cb0680-31b1-11eb-87ff-745d017b0a17.png)

# 4 Transformer architectures 

트랜스포머의 motivation은 모델을 **더 빨리** 학습시켜 더 큰 모델을 만드는 것에 있다. RNN 계열의 아키텍처는 recurrent한 속성이 있기 때문에 parallel한 연산을 수행할 수가 없다. 그럼에도 불구하고 이러한 구조는 반드시 필요한데, **long sequence length** 문제를 해결할 수 있는 건 attention mechanism이기 때문이다. Attention이 하는 일은 어떠한 state에도 접근할 수 있게 하는 것이고, 이것만 따로 구현할 수 있다면 RNN 구조에서 벗어날 수 있을 것이다. 이러한 아이디어가 바로 transformer의 탄생 배경이 된다.

![](https://user-images.githubusercontent.com/47516855/100516335-90f20d80-31c6-11eb-95a0-c7202e4d3ba7.png)

[Attention is all you need. 2017. Aswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin](https://arxiv.org/pdf/1706.03762.pdf)

Transformer구조는 attention은 유지하되 recurrent는 벗어나는 형태를 취하고 있다. 이 구조는 machine translation을 위해 제안되었는데, 복잡한 encoder/decoder를 갖고, 많은 attention distribution을 만들어서 이를 수행한다.

Transformer는 모든 곳에 attention을 거는데, 여러 attention 중에 가장 간단한 형태인 dot-product를 사용한다. Input으로 query와 key-value 쌍을 받아 query와 key간의 similarity를 계산하고, 이에 대응하는 value와의 attention을 계산한다. 이 결과로 value의 weighted sum (query-key)형태로 output이 나오게 된다.

![](https://user-images.githubusercontent.com/47516855/100519968-98bdac00-31de-11eb-9115-089d6216b336.png)

만일 key의 dimension d_k가 커지게 된다면, q와 k의 dot product의 값이 커질 것이고, 이로 인해 softmax내의 값이 커지는 효과가 일어난다 (high variance). 따라서 softmax을 통한 gradient가 매우 작아지게 되므로 학습이 잘 안되는 효과가 일어난다. 이를 방지하기 위해 sqrt(d_k)로 나눠준다.

> $$
> \text{var}(q^Tk) = \text{var}(\sum u_i v_i) =  k\text{var}(x_1y_1) = k\text{E}[x_1^2][y_1^2]=k\sigma^4
> $$
>
> <img src="https://render.githubusercontent.com/render/math?math=\text{var}(q^Tk) = \text{var}(\sum u_i v_i) =  k\text{var}(x_1y_1) = k\text{E}[x_1^2][y_1^2]=k\sigma^4">

![](https://user-images.githubusercontent.com/47516855/100520102-6f515000-31df-11eb-97b2-5e77e219e872.png)

> softmax with small gradient에 대한 추가 설명:
>
> softmax의 값이 크면, 특정 노드의 softmax의 값은 1에 가까워 지게 된다.
>
> ```python
> v1 = np.array([0.3, 0.4])
> print(np.exp(v1)/np.sum(np.exp(v1)))
> 
> array([0.47502081 0.52497919]) # 고르게 분산
> 
> v2 = v1 * 100
> print(np.exp(v2) / np.sum(np.exp(v2)))
> 
> array([4.53978687e-05 9.99954602e-01]) # 1에 가까운 값
> ```
>
> softmax의 derivative를 계산하면 다음과 같게 된다.
> $$
> \cfrac{\partial y_i}{\partial z_j} = 
> \begin{cases}
>  y_i(1 - y_i) \text{ if  }  i = j \\
>  y_i(y_j) \text{ if  }  i \neq j\
> \end{cases}
> $$
> <img src="https://render.githubusercontent.com/render/math?math=\cfrac{\partial y_i}{\partial z_j} = 
> \begin{cases}
>  y_i(1 - y_i) \text{ if  }  i = j \\
>  y_i(y_j) \text{ if  }  i \neq j\
> \end{cases}">
>
> i와 j가 같을 때 (정답) 0과 1값에서 최솟값이 되고, i와 j가 다른경우 (오답) y_i는 큰 값인데 반해 y_j는 0에 가깝게 된다. 따라서 gradient가 작게된다.

또 다른 중요한 점으로는 multi-head attention이 있다. 만일 하나의 attention만 사용하게 될 경우 우리는 한 가지 방법으로만 **attend**하게 된다. 예를 들어 dependency parser를 구축한다고 생각해보자. 우리는 headword도 파악해야 하지만, 이의 dependent word도 파악해야 할 것이다. 따라서 multi-head attention을 도입하여 attend를 다양한 측면에서 하도록 구성한다. 

![](https://miro.medium.com/max/437/1*5h3HHJh7kgezyOdTcRZc0A.png)

multi-head attention의 결과는 이후 input vector와의 residual을 통해 합쳐지고, layer normalization을 수행한다. 이후 2개의 FC와 ReLU를 통과하고 앞서 해주었던 Add & Norm을 수행하게 된다.

![image](https://user-images.githubusercontent.com/47516855/100523084-ecd18c00-31f0-11eb-9851-5712ec3cedd1.png)

이러한 작업을 총 6번 수행해주게 된다. 이러한 구조는 처음에 문장에 대해서 attention을 수행하여 정보를 수집하고, 다음 transformer block에 이를 전달해주는 것으로 생각할 수 있다.

한 가지 흥미로운 점은 이 모델이 언어 구조에서 흥미로운 것을 잘 attend한다는 것이다. 다음 그림을 보면 making이 more과 difficult를 attend하는데, 이는 argument와 modifier임을 확인할 수 있다.

![](https://user-images.githubusercontent.com/47516855/100523496-3b345a00-31f4-11eb-8c7d-08183af60861.png)

또한, pronoun의 경우 이의 modifier(application) 뿐만 아니라 reference에 attend하는 것을 볼 수 있다.

![](https://user-images.githubusercontent.com/47516855/100523574-d6c5ca80-31f4-11eb-91fe-95080d518ebd.png)

이후에는 decoding을 수행하게 된다. 강의에서는 그냥 넘어감으로 간단하게만 설명해보면, 디코더에서는 sub-sequence를 봐선 안되므로 (즉, 한국어를 받아서 영어를 해석해야 하는데, sub-sequence를 본다는 것은 도착어가 무엇인지를 이미 안 다는 뜻이나 다름없다) 이에 대해 masking을 씌어주고 이후 단어가 아닌 이전 단어들에만 attend하게 한다. 그리고 encoder (key, value)와 decoder (query)의 attention을 통해 translation과정에서 어떠한 source attention에 attend해야 하는지를 파악하게 된다.

![](https://user-images.githubusercontent.com/47516855/100523686-bfd3a800-31f5-11eb-94aa-1ea70384e532.png)

Details (in paper and/or later lectures):

- Byte-pair encodings
- Checkpoint averaging
- ADAM optimizer with learning rate changes
- Dropout during training at every layer just before adding residual 
- Label smoothing
- Auto-regressive decoding with beam search and length penalties 

Use of transformers is spreading but they are hard to optimize and unlike LSTMs don’t usually just work out of the box and they don’t play well yet with other building blocks on tasks.

# 5. BERT

[BERT (Bidirectional Encoder Representations from Transformers):Pre-training of Deep Bidirectional Transformers for Language Understanding)](https://arxiv.org/abs/1810.04805)

BERT는 가장 최근에 (현재는 GPT-3) 개발됐으면서 가장 좋은 성능을 보이고 있는 모델이다. 이는 트랜스포머의 인코더만을 이용하는 모델로 모든 task 기록을 갈아치우며 SOTA가 됐다. 여기에는 새롭고 흥미로운 아이디어가 있는데, 전통적인 LM은 unidirectional이고 잘 작동하긴 하지만, 양쪽 방향에서 보는 bi-directional이 제공하는 contexct와 meaning이 없다는 단점이 있다. 그렇지만 bi-directional하면 단어가 자기 자신을 참고하는 **crosstalk**문제가 발생한다. 

![](https://user-images.githubusercontent.com/47516855/100523993-df6bd000-31f7-11eb-81e9-b8151133d8ae.png)

BERT는 단어에 **mask**를 씌우는 방법으로 이를 해결했다. 예를 들어 *the man went to the store to buy a gallon of milk*가 있을 때, 이를 *the man went to the [MASK] to buy a [MASK] of milk* 로 masking하여 단어를 예측하게 하는 것이다. 이렇게 되면 crosstalk가 일어나지 않게 된다. 이러면 이 language model은 더 이상 sentence의 확률을 생성하는 모델이 아니게 되고, 그냥 단순히 빈칸 채우기가 된다. mask를 씌우는 비율은 trade-off인데, 15%로 논문에선 설정했다. 너무 크면 computationally expensive하고, 너무 작으면 context를 잘 이해하지 못한다.

Transformer 구조를 이용한 GPT같은 경우에는 traditional language model이고, ELMo의 경우엔 일종의 bidirectional language model이라고 볼 수 있다. 다만, 이 둘은 따로 학습이 되고, 나중에 concatenation한다. BERT는 이 둘의 장점을 고스란히 살렸고, 덕분에 좋은 결과를 얻었다.

![image](https://user-images.githubusercontent.com/47516855/101906779-df5acf80-3bfc-11eb-81b4-23b867e5945a.png)

또한, next sentence prediction (NSP)를 도입하였는데, 문장 두 개를 준 뒤 뒤의 문장이 실제로 연결되는 문장인지, 아닌지를 파악하는 작업을 모델이 학습하게 했다. 이는 도움이 되긴 하지만 필수적인 작업은 아니다. 이러한 작업은 QA, NLI 같은 문제에서 유용하다.

다음 그림은 embedding 과정을 시각화 한 것이다. 각 각의 단어에 대해 총 세 개의 임베딩을 합치게 되는데, 하나는 토큰 임베딩 (노란색), 하나는 포지셔널 임베딩 (흰색), 그리고 앞서 설명한 NSP를 위해 문장을 구분해주는 segment embedding이 있다. 

![](https://user-images.githubusercontent.com/47516855/100524309-615cf880-31fa-11eb-9513-f7833987f02a.png)

그 후 pre-trained model 위에 fine-tuning head를 붙여 작업을 수행하게 된다. 이는 앞서 설명했던 ULMfit처럼 모델의 구조를 그대로 유지할 수 있다는 장점을 갖는다.

![image](https://user-images.githubusercontent.com/47516855/101907559-11b8fc80-3bfe-11eb-8865-37c89a892c54.png)

이후로는 모델 performance에 대해 설명하고 있으므로 패스.





  