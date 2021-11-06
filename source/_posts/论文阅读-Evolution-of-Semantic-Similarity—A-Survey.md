---
title: 论文阅读-Evolution of Semantic Similarity—A Survey
date: 2021-10-11 17:19:04
tags: [论文]
---

#### Evolution of Semantic Similarity—A Survey

语义相似模型分为：

- knowledgebased， 基于知识
- corpus-based, 基于语料库
- deep neural network–based methods， 基于深度神经网络
- hybrid methods， 混合方法

Semantic Textual Similarity （STS）主要给出两段文本的相似度排名或百分比。

下图为文章介绍的相关模型

![image-20211011173454520](/Users/jiayi/Library/Application%20Support/typora-user-images/image-20211011173454520.png)

##### 基于知识的模型

基于知识的方法计算两个术语之间的语义相似度基于一个或多个知识库，如本体/词汇数据库，叙词表，词典等。

常用的知识库一般是词汇数据库lexical databases：

- WordNet：wordNet用node代表词义，并通过edge定义词之间的关系，两个词之间的相似度，取决于它们之间的距离。

- Wikipedia：基于Wikipedia，每个词组都有一个文章页面与之相关联，每篇文章同时具有标题、neighbors、描述和分类，这些作为特征都可以用于相似性的计算。
- BabelNet：babelNet是一个集合了WordnNet以及Wikipedia的词汇数据库

基于知识模型方法主要包括以下类型：

- Edge-counting 边计数法

  *Path*方法：

  ​	将词抽象为图节点，根据词的分类将两个词连接起来，然后统计边的数量作为两个term的相似度。
  $$
  sim_{path} (t1,t2) = \frac{1}{1 + min_len(t1,t2)}
  $$
  

  *wup*方法：

  ​	该方法统计每个term自身的分类深度，以及两个term（Least Common Subsumer）最近祖先分类点的深度：
  $$
  sim_{wup} (t_1,t_2) = \frac{2depth(t_{lcs})}{depth(t_1) + depth(t_2)}
  $$
  ​	

  缺点：忽略了本体中的edges不必等长的事实

- Feature-based 特征法

  基于词汇的属性，如注解、相似概念等，进行相似度的计算。

  *Lesk*：衡量两个词的相似度，主要考虑两个词的注解以及在WordNet中表示该词含义的词的注解的overlap

- Information Content-based 基于信息内容的方法

- Combined Knowledge-based 

##### 基于语料库的STS

基于语料库的方法根据从大规模的语料库中抽取出的信息，来衡量两个term之间的相似度。词的实际意义并没有被作为依据，而是基于一种"分布假设"，即相似的词语出现在一起 “similar words occur together, frequently”。

基于分布假设，出现了很多techniques，来构建文本数据的向量表示。

词嵌入提供了词的向量表示，在向量中，保持了词语之间的语义关系，常见的词嵌入包括：

- *word2vec*
- *GloVe*
- *fastText*
- *BERT*

在使用词嵌入作为相似度的衡量标准时，不可避免地会遇到“ Meaning Conflation Deficiency”词融合缺陷，举个例子，"finance"与"river"在语义空间中的位置很近，因为它们的一个相关词“bank”具备多重含义。

关键是要认识到，词嵌入本身是基于大规模语料库的，因此它的分布假设等与采取的语料库息息相关。

1. 基于语料库的方法：
   - LSA（Latent Semantic Analysis）
   - HAL（Hyperspace Analogue to Language）
   - ESA（Explicit Semantic Analysis）
   - Word-alignment Models
   - LDA（Latent Dirichlet Allocation）
   - Normalized Google Distance 
   - Dependency-based Models
   - Kernel-based Models
   - Word-attention Models

##### 深度神经网络方法

- DAM（Decomposable Attention Model）
- Transformer-based models

##### 混合方法Hybrid

- NASARI
- MSSA
- UESTS

