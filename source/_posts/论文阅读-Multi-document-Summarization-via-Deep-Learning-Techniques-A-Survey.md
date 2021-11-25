---
title: '论文阅读-Multi-document Summarization via Deep Learning Techniques: A Survey'
date: 2021-11-08 17:07:02
tags: [论文]
---

## Multi-document Summarization via Deep Learning Techniques: A Survey

*基于深度学习的多文档摘要模型综述*

**Computation and Language. 2020. [Congbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+C), [Wei Emma Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W+E)**

```
@misc{ma2020multidocument,
      title={Multi-document Summarization via Deep Learning Techniques: A Survey}, 
      author={Congbo Ma and Wei Emma Zhang and Mingyu Guo and Hu Wang and Quan Z. Sheng},
      year={2020},
      eprint={2011.04843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### **Summary**



### **Research Objective(s)**

作者的研究目标是什么？



### **Background / Problem Statement**

研究的背景以及问题陈述：作者需要解决的问题是什么？



### **Method(s)**

多文档摘要的目标是从一批文档集合D中生成一个简洁且信息丰富的摘要Sum。文档集合D是主题相关的一系列文本。

The aim of multi- document summarization is to generate a concise and informative summary 𝑆𝑢𝑚 from a collection ofdocuments𝐷.𝐷denotesaclusteroftopic-relateddocuments{𝑑𝑖 |𝑖∈[1,𝑁]},where𝑁isthe number of documents. Each document 𝑑𝑖 consists of 𝑀 sentences 􏰈𝑠𝑖,𝑗 | 𝑗 ∈ [1, 𝑀]􏰉. 𝑠𝑖,𝑗 refers to the 𝑗-th sentence in the 𝑖-th document. 

作者对多文档摘要的流程进行了总结与梳理，通常包括1⃣️选择一个合适的文档连接方法2⃣️对文档进行预处理3⃣️通过深度学习模型获取语义丰富的表达4⃣️融合各种类型的表达，5⃣️句子选择或摘要生成

The first step is to select an appropriate concatenation approach for input documents. The second step is pre-processing these documents, such as segmenting sentences, tokenizing non-alphabetic characters and removing punctuations [118]. Then, an appropriate deep learning based model is chosen to generate semantic rich representation for downstream tasks. The next step is to fuse these various types of representation for later sentence selection or summary generation. Finally, through these five steps, multiple documents are transformed into concise and informative summaries.

![image-20211108193513722](https://tva1.sinaimg.cn/large/008i3skNgy1gw7z2428ezj312u0aojsn.jpg)

#### input document types 输入文档类型

- 多篇短文档 Many short documents.

  每篇文档长度相对较短，但数量很大，如产品评论生成。

- 少量长文档 Few long documents.

  每篇文档长度相对较长，但数量较少，如新闻摘要生成。

- 混合文档类型 Hybrid documents. 

  一篇或多篇长文档同时伴随着若干篇短文档，如新闻稿+新闻短评，论文+简短的引用。

#### Concatenation Methods 连接方法

1. Flat Concatenation 平滑连接

   将所有输入文档囊括在一个输入当中，并当作一个整体的sequence进行处理，简单但powerful。

    All input documents are spanned and are processed as a flat sequence.

2. Hierarchical Concatenation 分层连接

   借助分层连接，有助于模型获得语义丰富的表示，从而提高模型的有效性。当前的分层连接方法，包含两种方案：

   - 文档级别：分别得到文档的摘要表示，这些表示在后续过程中进行融合。

     For the document-level concatenation methods, a condense model [3] is proposed to learn document-level representation separately in a cluster and these representation are fused in the subsequent processes.

   - 单词/句子级别：包含聚类与关系图方法

     **聚类方法：**首先对输入对文档进行句子级聚类，然后从不同的聚类中选择句子，保证最多从一个聚类中选择出一个句子，从而减少冗余信息，并提高信息的覆盖率。

     First allowing the model group related sentences. Then, the model selects sentences from diverse clusters, and at most one sentence will be selected from a cluster. By doing so, it will decrease redundancy and increase the information coverage for the generated summaries.

     **关系图方法：**构造句子关系图来表示文档之间的层次关系，常用的构造方法有余弦相似图、近似语篇图以及个性化语篇图。还有一种异构图模型利用单词构造句子与句子、句子与文档之间的关系图。

     Cosine similarity graph, approximate discourse graph, and personalized discourse graph are the most commonly used methods recently for building sentence graph structures. The heterogeneous graph model leverages words as intermediate nodes to construct a document-document, sentence-sentence and sentence-document hierarchical structure.

     

   ![image-20211108181754273](https://tva1.sinaimg.cn/large/008i3skNgy1gw7wtp5l44j310w0u00x5.jpg)

#### Summarization Construction Types 摘要构造方法

1. abstractive summarization 生成式摘要
2. extractive summarization 抽取式摘要
3. hybrid summarization 混合方法摘要

![image-20211108192305082](https://tva1.sinaimg.cn/large/008i3skNgy1gw7yphbejuj313w0fe0wb.jpg)

作者提出了一种概括模型结构的新方法，将模型结构分为了以下类型：

其中，绿色虚线box可以由其它网络模型灵活代替，蓝色实线box则表明通过神经网络或启发式设计方法进行的嵌入操作，可以是句子/文档等类型的表达。

 In this figure, deep neural models are boxed in green dotted line, which can be flexibly substituted by other backbone networks. The blue solid line boxes indicate the neural embeddings processed by neural networks or heuristic-designed approaches. It can be sentence/document representation or other types of representation. 

- Naive Networks

  朴素网络，DNN model作用是一个特征提取器，得到的representation交由下游做句子选择或摘要生成。

- Ensemble Networks.

  集成网络，输入docuemnts至多个模型，之后各模型得到的表达会进行融合来提高模型整体的表达能力。主流融合方法有投票或均值。

  Ensemble networks feed input documents to multiple paths with different network structures or operations. Later on, these representations are fused to enhance model expression capability. The majority vote or average can be used to determine the final solution.

- Auxiliary Task Networks

  

- Reconstruction Networks

  

- Fusion Networks. Fusion networks 

  

- Graph Neural Networks

  

- Hierarchical Networks

  文档集连接后输入第一层DNN后，获得其表层特征，之后将该第一层输出作为第二层DNN的输入，生成深层次表征，分层模型可以更有效地提取抽象层次和语义特征。

  The Hierarchical networks empower the model with the ability to capture abstract-level and semantic-level features more efficiently.

![image-20211108194035375](https://tva1.sinaimg.cn/large/008i3skNgy1gw7z7p99rdj310p0u0aes.jpg)

#### RNN based Models

#### CNN based Models

#### GNN based Models

自然语言数据通常由关系密切的词汇和短语组成，相比于序列的表示方法，图结构能够更好地进行表示。

Natural language data consist of vocabularies and phrases with strong relations and they can be better represented with graphs rather than in sequential orders

1. 基于GCN的方法^[1]^ ：构建句子关系图，送入GCN获得句子相关特征。

   This model first builds a sentence-based graph and then feeds the pre-processed data into a GCN [60] to capture the sentence-wise related features.

   **构造图的方法：**每个句子视作一个结点，句子间的关系为边，关系有余弦相似度、近似语篇、个性化语篇等。

   Defined by the model, each sentence is regarded as a node and the relation between each pair of sentences is defined as an edge. Inside each document cluster, the sentence relation graph can be generated through cosine similarity graph [32], approximate discourse graph [23] and the proposed personalized discourse graph. 

   句子关系图以及RNN抽取出embedding，都送入图卷积神经网络来得到句子最终的表达。最后将输出送入**文档GRU**完成集群嵌入，完全聚合句子之间的特征。

   Both of the sentence relation graph and sentence embeddings extracted by a sentence-level RNN are fed into graph convolution networks to produce the final sentence representation. With the help of a document-level GRU, the model generates cluster embeddings to fully aggregate features between sentences.

   ![image-20211109201016184](https://tva1.sinaimg.cn/large/008i3skNgy1gw95ox3f02j31440f4q4w.jpg)

2. 基于异构图的方法^[3]^ ：上面的方法只利用了句子级的关系，没有充分考虑单词、句子以及文章之间的关系。

   The existing graph neural networks based models are mainly focused on the relationship between sentences and do not fully consider the relations among words, sentences and documents

    HeterDoc-Sum Graph^[3]^是一种基于异构图注意力网络，它包含了单词节点、句子节点以及文章节点，句子节点和文章节点通过共现单词关系相连接

   Sentence nodes and document nodes are connected according to the contained word nodes. 

   TF-IDF值作为单词-句子以及单词-文章的权重

   三种关系图送入图注意力网络^[2]^中进行权重更新，每次更新时，对单词句子和单词文档进行双向更新，以更好地聚合跨层语义知识。

   ![image-20211110195359995](https://tva1.sinaimg.cn/large/008i3skNgy1gwaau91zzlj30pk0qkjv6.jpg)

#### PGNet based Models

Pointer-generator 网络被提出来解决摘要领域的事实错误以及高冗余性问题。

Pointer-generator networks is proposed to overcome the problems of factual errors and high redundancy in the summarization task. 

这类网络架构受到了pointer network、copynet、forced-attention sentence compression 以及 coverage mechanism的启发。对这些概念进行介绍：

**pointer network**



**copynet**



**converage mechanism**



**MMR（Maximal Marginal Relevance）**
$$
MMR = ArgMAX_{Di\ in \ 未选中集合}[\lambda Sim_1(D_i,Q) - (1-\lambda)max_{Dj \ in \ 已选中集合}Sim_2(Di,D_j))]
$$
*其中Q为用户，前半部分在摘要领域是对句子的打分。*

该方法目的在于减少排序结果的冗余，保证结果的多样性，常用于推荐领域。

More accurately, the MMR scores are multiplied to the original attention weights. MMR method is designed to select a set of salience sentences from source documents by considering both importance and redundancy indexes

#### Encoder-decoder based Models

**Encoder：**对原文档进行编码表示，编码包含了压缩语义和句法信息

**Decoder：**对encoder的编码结果进行处理，生成目标摘要

For multi-document summarization, the encoder embeds source documents into the hidden representations, i.e., word representation, sentence representation and document representation. Then, the representation containing compressed semantic and syntactic information is passed to the decoder to generate the target summaries. 

![image-20211110205341614](https://tva1.sinaimg.cn/large/008i3skNgy1gwackd5wj8j312g0ckabf.jpg)

#### Variational Auto-Encoder based Models

变分自编码器：



![image-20211110205840492](https://tva1.sinaimg.cn/large/008i3skNgy1gwacpjz9ulj315s09wgn2.jpg)

#### Transformer based Models



#### Deep Hybrid Models



### **Evaluation**

作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？



### **Conclusion**

作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?



### **Notes**

(optional) 不在以上列表中，但需要特别记录的笔记。



### **References**

[1] Michihiro Yasunaga, Rui Zhang, Kshitijh Meelu, Ayush Pareek, Krishnan Srinivasan, and Dragomir R. Radev. 2017. Graph-based Neural Multi-Document Summarization. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017).

[2] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph Attention Networks. arXiv preprint arXiv:1710.10903.

[3] Danqing Wang, Pengfei Liu, Yining Zheng, Xipeng Qiu, and Xuanjing Huang. 2020. Heterogeneous Graph Neural Networks for Extractive Document Summarization. arXiv preprint arXiv:2004.12393.