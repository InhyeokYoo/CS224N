# Introduction

CS224N 스터디의 학습내용 정리와 issue공유를 위한 repo입니다. 

- Stanford cs224n: Natural Language Processing with Deep Learning (**Winter 2019**)
  * [youtube](https://youtu.be/8rXD5-xhemo)
  * [syllabus, Winter 2019](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) // youtube version (기록 목차)
  * [syllabus, Winter 2020](http://web.stanford.edu/class/cs224n/)

- 📚 **References:**
  - Dan Jurafsky and James H. Martin. [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
  - Jacob Eisenstein. [Natural Language Processing](/src/eisenstein-nlp-notes.pdf)
  - Yoav Goldberg. [A Primer on Neural Network Models for Natural Language Processing](/src/A-Primer-on-Neural-Network-Models-for-Natural-Language-Processing.pdf)
  - Ian Goodfellow, Yoshua Bengio, and Aaron Courville. [Deep Learning](http://www.deeplearningbook.org/)
  - PRML 한국어 번역/정리: http://norman3.github.io/prml/


# Participant (alphabetical order)

| 이름 | repo |
| :---: | :---: |
|유인혁|[https://github.com/InhyeokYoo](https://github.com/InhyeokYoo) |
|장건희|https://github.com/ckrdkg |
|최슬기|[github](https://github.com/abooundev)  |


# 기록 (20.10.13 - )

![계획표](https://user-images.githubusercontent.com/47516855/97106526-8b0b9700-1705-11eb-8503-916730dcc116.png)


💡 **Importatn Note:**
- readme에 대한 관리는 제가 기본적으로 하겠지만, 각자의 작업물을 push하고 관리하는 건 각자 부탁드립니다.
- 매주 스터디에서 나온 질문/궁금증 등은 issue로 남기고 label을 달아주세요.
- 업데이트 하기 전에 fetch/pull 부탁드립니다.
- 양식의 통일 부탁드립니다 (markdown 추천)
  - format 추후 협의
  - naming convention
    - 강의 이름으로 된 파일을 올릴 것
    - 파일 이름 공백은 `-`로 대체하여 올릴 것
    - E.g. *Introduction-and-Word-Vectors.md*
- 수식 남기는 방법:
  - `<img src="https://render.githubusercontent.com/render/math?math={w_1, ..., w_m}">`에서 `math=`이후에 작성.
  - 결과: <img src="https://render.githubusercontent.com/render/math?math={w_1, ..., w_m}">
- Note이쁘게 만드는 방법:

  ```markdown
  | :exclamation:  This is very important   |
  |-----------------------------------------|
  ```
  - 결과

    | :exclamation:  This is very important   |
    |-----------------------------------------|

📑 **강의 목차:**  
| Lecture | 2019 | 2020 | 일치여부 | 발표자 | 발표날짜 | 링크 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
1 |  Introduction and Word Vectors/Gensim word vectors example |  | O | **유인혁** | 2020.10.13 | [Link](/week1) |
2 | Word Vectors 2 and Word Senses |  | O | **유인혁** | 2020.10.13 | [Link](/week1) |
3 | Word Window Classification, Neural Networks, and Matrix Calculus  | Word Window Classification, Neural Networks, and **PyTorch** | X |  **최슬기** | 2020.10.21 | Link |
4 | Backpropagation and **Computation Graphs** | Matrix Calculus and Backpropagation | X |  **장건희** | 2020.10.21 | | Link |
5 | Linguistic Structure: Dependency Parsing  |  | O | **유인혁** | 2020.11.01 | [Link](/week3/Linguistic-Structure-Dependency-Parsing.md) |
6 | The probability of a sentence? Recurrent Neural Networks and Language Models | | O | **최슬기**  | 2020.11.01 |  Link |
7 | Vanishing Gradients and Fancy RNNs | | O | **유인혁**  | 2020.11.04 |  [Link](/) |
8 | Machine Translation, Seq2Seq and Attention | | O | **장건희**  | 2020.11.04 |  [Link](/) |
9 | Practical Tips for Final Projects | | O | **장건희**  | 2020.11.11 |  [Link](/) |
10 | Question Answering and the Default Final Project | Question Answering, the Default Final Project, **and an introduction to Transformer architectures** | X | **최슬기**  | 2020.11.11 |  [Link](/) |
11 | ConvNets for NLP |  | O | **최슬기**  | 2020.11.18 |  [Link](/) |
12 | Information from parts of words: Subword Models | | O | **유인혁**  | 2020.11.18 |  [Link](/) |
13 | Modeling contexts of use: Contextual Representations and Pretraining | Contextual Word Representations: BERT (guest lecture by Jacob Devlin) | X | **유인혁**  | 2020.11.25 |  [Link](/) |
14 | Transformers and Self-Attention For Generative Models | Modeling contexts of use: Contextual Representations and Pretraining. ELMo and BERT | X | **장건희**  | 2020.11.25 |  [Link](/) |
15 | Natural Language Generation | | O | **장건희**  | 2020.12.02 |  [Link](/) |
16 | Reference in Language and Coreference Resolution | | O | **최슬기**  | 2020.12.02 |  [Link](/) |
17 | Multitask Learning: A general model for NLP? (guest lecture by Richard Socher) | Fairness and Inclusion in AI (guest lecture by Vinodkumar Prabhakaran) | X | **최슬기**  | 2020.12.02 |  [Link](/) |
18 | Constituency Parsing and Tree Recursive Neural Networks  | | O | **유인혁**  | 2020.12.02 |  [Link](/) |
19 | Safety, Bias, and Fairness (guest lecture by Margaret Mitchell) | Recent Advances in Low Resource Machine Translation (guest lecture by Marc'Aurelio Ranzato) | O | **유인혁**  | 2020.12.09 |  [Link](/) |
20 | Future of NLP + Deep Learning   | Analysis and Interpretability of Neural NLP | O | **장건희**  | 2020.12.09 |  [Link](/) |
