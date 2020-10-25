# Week 3 (20.10.28)

발표자: **유인혁**

이번 주제는 Language Modeling (LM)이다. 전통적인 확률적 LM인 N-gram부터 deep learning을 활용한 LM까지 전부 살펴보겠다.

# 1. Language Models

## 1.1 Introduction

- Language model은 특정한 sequence에서의 words가 등장 확률을 구하는 것이다
    - m개 단어 시퀀스 <img src="https://render.githubusercontent.com/render/math?math={w_1, ..., w_m}">의 확률을 <img src="https://render.githubusercontent.com/render/math?math=P{w_1, ..., w_m}">로 표현
- 


수식 테스트

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

<img src="https://render.githubusercontent.com/render/math?math=p(w_1 , ..., w_n) = \prod ^{i=m} _{i=1} P(w_i | w_1, ..., w _{i-1} \approx \prod ^{i=m} _{i=1} P(w_{i-n} | w1, ..., w _{i-1})">

노트 테스트
| :exclamation:  This is very important   |
|-----------------------------------------|

- 리스트 안에서 사용해보기

    | :exclamation:  This is very important   |
    |-----------------------------------------|