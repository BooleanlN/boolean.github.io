---
title: 论文阅读-Heterogeneous Graph Neural Networks for Extractive Document Summarization
date: 2021-11-15 21:00:57
tags: [论文]
---

## Heterogeneous Graph Neural Networks for Extractive Document Summarization

*Danqing Wang∗, Pengfei Liu∗ ACL. 2020*

```
@misc{wang2020heterogeneous,
      title={Heterogeneous Graph Neural Networks for Extractive Document Summarization}, 
      author={Danqing Wang and Pengfei Liu and Yining Zheng and Xipeng Qiu and Xuanjing Huang},
      year={2020},
      eprint={2004.12393},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[代码地址](https://github.com/dqwang122/HeterSumGraph)

### **Summary**

作者首次提出一种用于抽取式文本摘要刻画句子间关系的异构图，图中包含了**词节点**（基本语义单元）、**句子结点**（以及多文档摘要时用到的文档节点），词节点通过**词嵌入**做初始化，句子节点则通过**CNN+BiLSTM**抽取特征向量，以词节点的**TF-IDF值**作为词节点与句子节点之间的边。

之后，首先对句子结点值，通过GAT + FNN 进行聚合更新，再用更新后的句子结点对词结点进行更新，最后，对句子进行分类（留或不留），并用Trigram Blocking防止冗余。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gwrgywuvjvj30lg0am3yy.jpg" alt="image-20211125162038154" style="zoom:50%;" />

结果证明了作者提出的异构图在抽取句子关系时的效果，同时，在CNN/DailyMail等数据集中，表现要优于那些没有使用预训练模型，如BERT的模型。（因为计算资源限制，作者未引入BERT）

### **Research Objective(s)**

在抽取式文本摘要领域，为了从一个文档中有效地抽取出值得作为总结的句子，需要对句子之间的关系进行刻画。作者第一个**在基于图的神经网络方法中引入了不同的节点类型来解决文本摘要任务**，并对引入其他类型节点的好处进行了分析。

we are the first one to introduce different types of nodes into graph-based neural networks for extractive document summarization and perform a comprehensive qualitative analysis to investigate their benefits. 



### **Background / Problem Statement**

文本摘要分为生成式与抽取式，在抽取式文档摘要领域，目标是能够从原始文档中抽取出有重大意义的句子。为了抽取句子，其中一个重要的步骤就是刻画句子间关系，常用的方法有LexRank^[1]^、TextRank^[2]^，基于Approximate Discourse Graph(ADG)^[3]^，基于Rhetorical Structure Theory（RST）^[4]^。但这些方法常常依赖于外部工具，并需要考虑错误传播问题。

Recently, some works account for discourse inter-sentential relationships when building summarization graphs, such as the Approximate Discourse Graph (ADG) with sentence personalization features (Yasunaga et al., 2017) and Rhetorical Structure Theory (RST) graph (Xu et al., 2019). However, they usually rely on external tools and need to take account of the error propagation problem.

尽管这些模型取得了不错的成绩，但如何为文本摘要构建一个有效的图结构仍然是一个亟待解决的问题。

### **Method(s)**

作者引入了更多语义单元作为额外的节点来丰富句子之间的关系。作者选择单词作为语义单元，句子与句子之间没有直接相连的边，这种图结构具有以下优点：

1. 不同的句子之间可以通过重叠词信息相互作用
2. 词节点从句子中聚合信息，并进行更新，而其他模型word embedding通常是不变的 
3. 通过多个消息传递过程，可以充分利用不同粒度的信息。
4. 作者提出的异构图是可扩展的，例如，可以为多文档摘要任务引入文档结点。

具体方法：

给定文档$D = \{s_1,s_2,...,s_n\}$，我们的目标是预测一系列标注结果$Y = \{y_1,y_2,...,y_n\}，y_i \in \{0,1\}$，其中，$y_i = 1$表示该句子$s_i$应该包含在结果中。

在作者构建的异构图中，基本语义节点作为中继节点，其他单元作为super 结点，super结点与它包含的基本结点相连接，连接的权重为它们关系的重要程度。

basic semantic nodes (e.g. words, concepts, etc.) as relay nodes and other units of discourse (e.g. phrases, sentences, documents, etc.) as supernodes. Each supernode connects with basic nodes contained in it and takes the importance of the relation as their edge feature. Thus, high-level discourse nodes can establish relationships between each other via basic nodes.

<img src="/Users/jiayi/Library/Application%20Support/typora-user-images/image-20211123231937172.png" alt="image-20211123231937172" style="zoom:50%;" />

如上图所示，作者提出的模型包括三部分：图初始化器、异构图层和句子选择器。

**图初始化器：**

Word Encoder：$X_w \in R^{m \times d_w }$，词嵌入，word embedding，$d_w$是向量维度

Sentence Encoder：$X_s \in R^{n \times d_s}$，$d_s$为向量维度，其中首先通过CNN获取局部n-gram特征$l_j$，再通过BiLSTM获取句子级别的特征$g_j$，将两者相练，得到最后的向量$X_{sj} = [l_j;g_j]$

Edge Initializer：使用TF-IDF值，作为边的初始权重。

**异构图层：**

使用GAT^[6]^（graph attention networks）对语义单元结点进行更新。

$z_{ij} = LeakyReLU(W_a[W_qh_i;W_kh_j;e_{ij}])$

$\alpha_{ij} = \frac{exp(z_{ij})}{\sum_{l \in N_i}exp(z_{il})}$

$u_i = \sigma(\sum_{j \in N_i}{\alpha_{ij}W_vh_j})$

在这里，作者采用**多头self-attention**：

$u_i = ||_{k=1}^K\sigma(\sum_{j \in N_i}{\alpha_{ij}W_vh_j})$

除此之外，增加了**残差连接**解决梯度消失的问题：

$h'_i = u_i + h_i$

在每层graph attention layer之后，引入位置前馈层（position-wise FFN），该层由两个线性变换组成，类似Transformer那样。

之后，如下图所示，句子节点由相连接的词节点通过GAT与FFN进行聚合更新。更新完句子节点后，词节点再由更新后的句子节点进行聚合更新。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gwq6ltdd3sj30pc0mewhl.jpg" alt="image-20211124133631315" style="zoom:50%;" />

最终，出现次数多的单词节点会有更高的degree，表明它可能是该文章的keyword，包含更多keyword的句子理所当然地具备更高的优先级作为输出句子。

**GAT：**

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gwq5eyy671j31240jywg8.jpg" alt="image-20211124125519304" style="zoom:50%;" />

单层Graph Attentional Layer输入节点特征向量集：$h = \{h_1,h_2,...,h_n\},h_i \in R^F$，其中，F为向量维度

每一层输出为一个新的节点特征向量集合：$h' = \{h'_1,h'_2,...,h'_n\},h'_i \in R^{F'}$，其中，F'为新的向量维度

具体来讲，Graph attentional layer会对输入特征向量进行**self-attention**：$e_{ij} = a(Wh_i, Wh_j)$，其中a是从$R^{F'} \times R^{F'} -> R$的映射，W是被所有$h_i$**共享**的$R^{F' \times F}$的权值矩阵。这样的操作会将注意力分配到图中的所有节点当中，这样做会丢失结构信息。GAT采用一种masked attention的方式，仅将注意力分配到节点i的邻节点集上。

$\alpha_{ij} = softmax(e_ij) = \frac{exp(e_{ij})}{\sum_{k \in N_i}exp(e_{ik})}$，其中，$N_i$是结点i的邻近结点集合。

在*Graph Attention Networks*中，作者采取了一层前馈网络作为a的实现：

$\alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j]))}{\sum_{k \in N_i}exp(LeakyReLU(a^T[Wh_i||Wh_k]))} $

之后，就可以求得$h'_i$：$h'_i = \sigma(\sum_{j \in N_i}{\alpha_{ij}Wh_j})$

此外，为了提高模型的拟合能力，作者引入了多头self-attention，具体实现即同时采用多个$W_k$，之后将结果进行合并或求均值。

$h_i = {||}_{k=1}^K \sigma(\sum_{j \in N_i}{\alpha_{ij}W^kh_j})$，这是连接方式，还可以通过均分等方式：

$h_i = \sigma(\frac{1}{K}\sum_{k=1}^K\sum_{j \in N_i}{\alpha_{ij}W^kh_j})$

**句子选择器：**

对句子节点进行分类，交叉熵损失函数作为训练目标。

使用**Trigram Blocking**^[5]^，一个MMR方法进行句子去重，对句子进行排序，去掉与前序句子有三叉重合的句子。

**多文档摘要：**

如下图所示，作者通过在异构图中，引入文档节点，通过word node，以**TF-IDF值**作为edge，可以建立起文档与文档之间的层次关系。

文档结点与句子结点的更新步骤是一样的，与句子结点不同的是，文档结点使用它**包含的句子结点的均值池化值**作为初始值。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gwrfezzk1hj30ou0qu42a.jpg" alt="image-20211125152651920" style="zoom:50%;" />

### **Evaluation**

**数据集**：**CNN/DailyMail**、**NYT50**、**Multi-News**（多文档摘要）

**word embedding**：Glove.300d

**optimizer:** Adam

**评价指标：**Rouge-1、Rouge-2、Rouge-L

**Baselines**：

作者用三种抽取句子间关系的方法进行实验，以此证明**HETERSUMGARPH**在句子关系抽取中的效果更加powerful。

- **Ext-BiLSTM **：通过CNN+BiLSTM得到的句子向量直接进行classification
- **Ext-Transformer**：通过Transformer结构对完全连接的句子学习其pairwise交互结构得到encoder

- **HETERSUMGARPH**

如下图所示是作者提出的模型在CNN/DailyMail的实验对比结果，可以看出相比于其它的句子关系抽取方法，**作者提出的方法效果更好**。

并且，作者提出的模型性能要**优于之前所有未以BERT为基础的摘要模型**，并证明了**Trigram blocking**对ROUGE评分具有很好的提升效果，

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gwrg6qk4e1j30na0o6n1k.jpg" alt="image-20211125155332812" style="zoom:50%;" />



### **References**

[1] Gu ̈nes Erkan and Dragomir R Radev. 2004. Lexrank: Graph-based lexical centrality as salience in text summarization. *Journal of artificial intelligence re- search*, 22:457–479.

[2] Rada Mihalcea and Paul Tarau. 2004. Textrank: Bring- ing order into text. In *Proceedings of the 2004 con- ference on empirical methods in natural language processing*, pages 404–411.

[3] Michihiro Yasunaga, Rui Zhang, Kshitijh Meelu, Ayush Pareek, Krishnan Srinivasan, and Dragomir Radev. 2017. Graph-based neural multi-document summarization. *arXiv preprint arXiv:1706.06681*.

[4] Jiacheng Xu, Zhe Gan, Yu Cheng, and Jingjing Liu. 2019. Discourse-aware neural extractive model for text summarization. *arXiv preprint arXiv:1910.14142*.

[5] Yang Liu and Mirella Lapata. 2019b. Text summariza- tion with pretrained encoders. In *Proceedings of the 2019 Conference on Empirical Methods in Nat- ural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3721–3731, Hong Kong, China. Association for Computational Linguistics.

[6] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. *arXiv preprint arXiv:1710.10903*.

