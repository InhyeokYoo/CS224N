# Lecture 16
# Coreference Resolution

### 1. Coreference Resolution이란?
텍스트 속에서 현실에 존재하는 entity를 찾아내는 것

![](![](2020-12-01-18-56-30.png).png)

ex)위 문장에서 같은 entity를 언급하는 것은 Barack Obama, his, he

### 2. Coreference Resolution in Two Steps
(1) Detect the mentions
"I Voted for Nader because he was most aligned with my values," she said

(2) Cluster the mentions
"(I) Voted for [Nader] because [he] was most aligned with {(my) values}," (she) said

### 3. Mentions Detection
mentions: span of text (referring to some entity)

##### (1) mentions의 종류
- Pronouns: I, your, it, she, him
- Named entities: People, places
- Noun phrases: "a dog", "the big fluffy cat stuck in the tree"

the big fluffy cat stuck in the tree 같은 경우는 complex한 mention임 
-> why? 다른 mention들도 그 안에 포함되있기 때문
tree도 mention이 됨

##### (2) How can we detect mentions?
- Pronouns -> part of speech tagger
(어떤게 명사고 동사인지 대명사인지)
- Named entities -> NER system
(ex) 버락 오바마)
- Noun phrases -> parser
(문장 구조를 찾고 명사 구조를 찾기 위해 parser가 필요,
다음 주에 constituency parser를 다룰 텐데 가장 빠르게 mentions을 찾는 방법이라 함)

##### (3) Are these mentions?
- It is sunny -> It은 start of the sentence이지 어떤 의미가 있는건 아님
- Every student -> 구체적인 언급이 아니기 때문에 mention이 아님
- No student -> 마찬가지로 아무 것도 의미하지 않음(존재하지 않기 때문)
- The best donut in the world -> 불명확, 
최고의 도넛 무엇이냐에 대한 논쟁이라면 reference가 있다고 말할 수 있지만 최고의 도넛을 찾고 있다면 reference가 존재하지 않는다.
- 100 miles -> quantity를 나타내는 것도 reference가 존재하지 않음

##### (4) How to deal with these bad mentions?
- bad mentions를 필터링하도록 classifier를 train, classifier는 mention된 것과 아닌 것을 분류

- 하지만 사람들은 위 단계를 skip하고 대부분 모든 candidate mentions를 찾음
-> why? 효과가 좋은 것으로 밝혀짐. 모든 mention을 찾은 후에 핵심적인 mention을 찾기 위해 clustering을 사용
if) 'No stuendt' 같은 mentnion이 나오면 다른 것들과 clustering 시키지 않음


##### (5) Can we avoid a pipelined system?
part of speech, NER, parser, named mention detector 등이 pipeline
이건 traditional한 방법, 2016년 pipeline이 등장하기 전까지 coreference system의 모든 system
파이프라인없이 coref cluster를 하는 end-to-end coreference system을 build할 수 있을까?
강의 마지막 부분에서 설명

### 4. On to Coreference! First, some linguistics

##### (1) 언어학 관점
![](2020-12-03-22-07-01.png)

(두 개의 언급이 세계의 동일한 실체를 언급할 때

anaphor란 다른 표현, 대개는 그 글에서 이전에 나왔던 표현에서 그 의미를 빌려 오는 것을 가리키는 용어)

어떤 텍스트에서 he라는 단어가 나오면 대용어 즉 anaphor다. anaphor인건 알겠는데 그가 누군지 알아야 한다. 그럼 텍스트를 보고 Barack Obama를 의미하는 것을 알게 된다. 

##### (2) Anaphora vs Coreference
모든 anaphoric relations가 coreferential한 건 아니다.
![](2020-12-03-22-48-34.png)
23

a concert와 The tickets는 anaphoric relationship이다.
But, 명백하게 다른 entity이므로 coreference relationship은 아니다.

##### (3) Cataphora
![](2020-12-03-22-55-18.png)
하지만 현대 언어학에서 cataphora는 사용되지 않는다.

##### (4) Four kinds of Coreference Models
![](2020-12-03-23-04-09.png)

### 5. Traditional pronominal anaphora resolution: Hobbs' naive algorithm

##### (1) 순서
1) 대명사가 있는 NP에서 시작
2) 시작위치에서 등장하는 첫번째 NP나 S로 이동 -> 해당 노드: X, X에 인접한 path: p라고 부름
3) X의 왼쪽 p부터 모든 가지를 left-to-right, breadth-first으로 이동
등장한 NP는 선행사가 됨
4) X가 가장 높은 S라면, 이전 문장의 tree를 3)과 같이 이동
X가 가장 높은 S가 아니라면 5)로 넘어감
5) X에서 바로 등장하는ㄴ NP나 S로 이동 -> 해당 노드: X, x에 인접한 path:p 라고 부름

##### (2) Example

![](2020-12-05-15-41-28.png)

(pronoun이 인 him이 저기 있으니까 그 위에 NP에서 부터 시작
시작위치에서 등장하는 S 즉 맨 위로 이동하여 여길 X라고 부르고 그 인접 경로를 p라고 부름
X의 왼쪽 p부터 쭉 순회, 여기서 등장하는 NP는 선행사가 됨
하지만 Stephen Moss는 선행사가 될 수 없음 말이 이상해짐
선행사가 되려면 Stephen Moss 와 S사이에 다른 무언가가 있어야함 ex)mother
그래서 X가 가장 높은 S라면 이전 문장 tree를 순회
NP Niall Ferguson이 선행사)

##### (3) Knowledge-based Pronominal Coreference
![](2020-12-05-16-21-06.png)

이 문장들은 각각 같은 구조를 가지고 있지만 it이 가리키고 있는것은 다르다. 이러한 경우에는 위에 설명한 Hobb's algorithm을 사용할 수 없다. 

### 6. Coreference Models: Mention Pair
![](2020-12-05-16-37-03.png)
![](2020-12-05-16-37-12.png)

##### (1) Mention Pair Training

![](2020-12-05-17-13-49.png)


 Binary Classifier의 결과값은, 두 mention이 coreferent 하다면 1을, coreferent 하지 않다면 -1값을 내보낸다.

p(mj, mi) = Coreferent할 확률을 예측하는 model
cross entropy loss를 사용하여 학습

##### (2) Mention Pair Test Time

1) 각 mentions를 pair로 묶어 classifier에 넣는다.
2) threshold를 기준으로 coreference link가 생긴다.
3) mention clustering을 위해 transitive closure라는 방법을 사용
- transitive closure: A가 B와 coreferent하고 B가 C와 coreferent 하다면 A와 C는 coreferent하다.
4) transitive closure 때문에 모든 mention이 하나로 묶일 수 있는 over cluster의 위험성이 존재한다.
5) singleton mentnion이 존재할 수도 있다.
singletion mention: coreferent를 이루지 않는 mention


##### (3) Mention Pair Models: Disadvantage
![](2020-12-05-19-34-03.png)
A long document with lots of mentions -> 해당하는 모든 mention을 찾는게 아니라 대표하는? 잘 표현하는? 특정 mention하나를 찾아낸다.

### 7. Coreference Models: Mention Ranking
![](2020-12-05-20-11-26.png)
- she와 다른 mention들을 각각 pair하여 softmax 적용
- 가장 높은 값의 mention만 coreference link

##### (1) Coreference Models: Training
![](2020-12-05-20-18-31.png)

목표: antecedents중에서 high score of coreference얻는 것

##### (2) Mention Ranking Models: Test Time
![](2020-12-05-20-20-57.png)
각 mention 당 하나의 antecedent를 제공하는 것 말고는 Mention Pair Model과 동일하다.

##### (3) How do we compute the probabilities?
###### A. Non-neural statistical classifier
![](2020-12-05-20-28-06.png)
classical한 방법
(the mining conglomerate, the company는 word2vec사용하여 유사성 평가할 수 있다.
)
위의 기능들을 statistical classifier에 넣는다.
2000년대 핵심 시스템

###### B. Simple neural network
![](2020-12-05-22-03-49.png)
word embedding과 categorical feature를 input으로 하여 score 계산

###### C. More advanced model using LSTMs, attention
![](2020-12-05-23-26-12.png)
목표: end-to-end coreference system의 생성
![](2020-12-05-23-50-58.png)

각 단어마다 word embedding을 한다. 
매트릭스와 캐릭터 레벨 CNN을 포함하는 단어를 연결하여 각 토큰을 나타낸다.
그 다음 스텝에서 Bidirectional LSTM을 실행한다.
그래서 그 이후에 스팬에 대한 표현을 한다.
이 스팬 표현은 세 부분으로 나눌 것입니다.
![](2020-12-05-23-58-41.png)

마지막꺼는 additional features라고 해서
speaker와 adress에 표시를 한다.
grammatical한 역할처럼 text안에서 발견할 수 없는 것을 표시한다.

if) the postal service일 때
the와 service는 BiLSTM을 사용한 span representation

여기서 좀 까다로운게 등장하는데 우린 text속에서 headword를 찾길 원한다. 그래서 attention을 사용한다.
어텐션을 내부 메커니즘으로 사용하여 대략적인 head를 만든다.

그리고 마지막으로 score를 낸다.
![](2020-12-06-00-18-21.png)

* Problem
![](2020-12-06-00-28-12.png)

모든 pair를 다 계산할 수 없기에 pruning 작업이 필요

### 8. Last Coreference approach: Clustering-Based
직접 사용해보자!
![](2020-12-06-00-37-53.png)
the company != the product

if you look at coreference resolution papers, there are many metrics. (MUC, CEAF, LEA, B-CUBED, BLANC)

![](2020-12-06-16-25-07.png)
B-cubed는 MUC의 대안으로 정의되었다. B-cubed에는 precision, recall 2가지 유형이 있으며, 두 가지 모두 각 precision score의 weighted averages입니다.

### Conclusion
![](2020-12-06-16-32-26.png)

![](2020-12-06-16-33-01.png)
