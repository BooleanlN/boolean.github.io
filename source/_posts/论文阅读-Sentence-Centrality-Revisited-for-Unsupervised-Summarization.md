---
title: 论文阅读-Sentence Centrality Revisited for Unsupervised Summarization
date: 2021-10-19 16:11:15
tags: [论文]
---

##  Sentence Centrality Revisited for Unsupervised Summarization（基于再访问句子中心性的无监督文本摘要生成）

*2019.6.8 Hao Zheng. Mirella Lapata. ACL 2019*

```
@misc{zheng2019sentence,
      title={Sentence Centrality Revisited for Unsupervised Summarization}, 
      author={Hao Zheng and Mirella Lapata},
      year={2019},
      eprint={1906.03508},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[code地址](https://github.com/ mswellhao/PacSum)

### 0. Summary 

作者在本文提出了一种无监督文本抽取摘要模型，相比于TextRank方法，PacSum主要做了两方面改进：

一是作者重新定义了句子中心性的计算方式，将句子之间的相对位置纳入了计算，改进了度中心性的计算方法。

二是通过BERT模型进行句子语义的捕获，这与文章提出时TextRank常采用的tf-idf在语义表示上有了提升（TextRank+BERT作者进行实验发现，效果比tf-idf要差，作者推测是因为TextRank模型错误的句子中心性导致的）

除此之外，作者基于句子分布假设，提出了与skip-thought vector不同的fine-tune方法，并取得了更好的效果。

作者在英文数据集、中文数据集进行了有无监督模型、有监督模型的实验，其中，取得了比文章提出时无监督模型更好的效果，并在英文数据集中，取得了与有监督模型十分相近的分数。

### 1. **Research Objective(s)**

为了解决文本摘要领域，难以为不同类型的摘要、领域、语言去获得或创建大规模的高质量训练数据的问题，作者提出一个普遍适用的图排序算法，并且从两个方面改进了句子节点中心性计算：1)   基于BERT模型，捕捉句子语义 2) 使用有向边构建句子节点图，使用有向边，去表明任意两个句子的中心性分别被他们相对位置所造成的影响。

it is unrealistic to expect large-scale and high-quality training data to be available or created for different types of summaries, domains, or languages. Author revisit a popular graph-based ranking algorithm and modify how node (aka sentence) centrality is com- puted in two ways: (a) we employ BERT, a state-of-the-art neural representation learning model to better capture sentential meaning and (b) we build graphs with directed edges arguing that the contribution of any two nodes to their respective centrality is influenced by their relative position in a document.

### **2. Background / Problem Statement**

**背景：**

- 基于神经网络的方法在成百上千的大规模数据集上取得了可观的效果

Modern neural network-based approaches have achieved promising results thanks to the availability of large scale datasets containing hundreds of thousands of document-summary pairs

- 大规模的、高质量的数据集难以创建或获得

it is unrealistic to expect that large-scale and high-quality training data will be available or created for different summarization styles

- 无监督方法是之前研究的重点，其中比较流行的是基于TextRank方法。

A very popular algorithm for extractive single-document summarization is TextRank (Mihalcea and Tarau, 2004)

**问题阐述：**

- 文本摘要领域，难以为不同类型的摘要、领域、语言去获得或创建大规模的高质量训练数据。

it is unrealistic to expect that large-scale and high-quality training data will be available or created for different summarization styles

- 无监督方法中，基于句子中心性的方式如TextRank可以从两方面改进，一是使用BERT模型获取文本语义，二是构建有向图来表示两个句子节点对相互之间的贡献是不同的

We employ BERT (Devlin et al., 2018), a neural representation learn- ing model which has obtained state-of-the-art re- sults on various natural language processing tasks including textual inference, question answering, and sentiment analysis

we advocate that edges should be *directed*, since the contribu- tion induced by two nodes’ connection to their re- spective centrality can be in many cases unequal.

### 3. **Method(s)**

**3.1 句子中心性计算**

**3.1.1 无向图计算方法**

A node’s centrality can be measured by simply computing its degree or running a ranking algorithm such as PageRank

结点中心性计算可以简单的通过度计算，也可以通过排序算法，如PageRank。

**度中心性**计算方法：
$$
centrality(si) = \sum_{j ∈{1,..,i−1,i+1,..,n}}{e_{ij}}
$$
**TextRank**计算方法：

度中心性计算方法只考虑局部连通性，而PageRank算法通过递归为所有节点分配了相对的分数，与高得分节点相连接的节点对分数贡献更大。

Whereas degree centrality only takes local connectivity into account, PageRank assigns rela- tive scores to all nodes in the graph based on the recursive principle that connections to nodes hav- ing a high score contribute more to the score of the node in question.

**3.1.2 有向图计算方法**

理论支持：RST（Rhetorical Structure Theory），修辞结构理论，表示， 语篇单元重要性和显著性是不同的。根据其文本重要性，分为核心句与附属句。

The idea that textual units vary in terms of their importance or salience, has found support in various theories of discourse structure including Rhetorical Structure Theory.

in terms of their text importance: *nuclei* denote central segments, whereas *satellites* denote peripheral ones.

计算文本核心性的方法：通过文本的相对位置进行近似计算，其中，文档中出现较早的句子应该更核心。

We instead approximate nuclearity by relative position in the hope that sentences occurring earlier in a document should be more central.

**方法实现**

给定任意来自同篇文档的句子$s_i$，$s_j$：
$$
centrality(s_i) = \lambda_1\sum_{j<i}{e_{ij}}  + \lambda_2\sum_{j>i}{e_{ij}}
$$
其中，$\lambda_1$与$\lambda_2$表示前向句与后向句的有向边权重系数。

 λ1, λ2 are different weights for forward- and backward-looking directed edges.

在实验中，我们设置$\lambda_1 + \lambda_2 = 1$控制超参数的数量

During tuning experiments, we set λ1 + λ2 = 1 to control the number of free hyper-parameters.

经过实验，我们发现前向系数$\lambda_1$趋向于负值，这表示与前面内容的相似性，实际上会损害其本身的中心性。

we find that the optimal λ1 tends to be negative, implying that similarity with previous content actually hurts centrality

未来可以通过PageRank等将负值边纳入计算。

Although it is possible to use some extensions of PageR- ank (Kerchove and Dooren, 2008) to take negative edges into account, we leave this to future work and only consider the definition of centrality from Equation (6) in this paper.

**3.2 语句相似度计算**

许多TextRank方法基于符号句表示（symbolic sentence representations）如tf-idf进行文本的表示。

There are many variations of the similarity function of TextRank (Barrios et al., 2016) based on symbolic sentence representations such as tf-idf.

本文采用一种神经网络分布式表示——BERT模型作为encoder，并通过一种句子级别分布假设对其进行fine-tune。

We use BERT (Devlin et al., 2018) as our sentence encoder and fine-tune it based on a type of sentence-level distributional hypothesis

**3.2.1 句子级分布式假设**

为了对BERT模型进行fine-tune，作者采用了一种句子级分布式假设来定义一个训练目标。

To fine-tune the BERT encoder, we exploit a type of sentence-level distributional hypothesis (Harris, 1954; Polajnar et al., 2015) as a means to define a training objective. 

与通过重构编码句子的邻近句的Skip-thought vectors不同的是，作者借用了单词分布式假设的负采样方法。

**损失函数：**
$$
log\sigma({v^{'}_{s_{i-1}}}^Tv_{s_{i}}) + log\sigma({v^{'}_{s_{i+1}}}^Tv_{s_{i}}) + E_{s ̃p{(s)}}[log\sigma({-v^{'}}^T{v_s})]
$$
其中 vs 和 vs′ 句子s在两个不同参数的BERT 编码器的不同表示， σ是sigmoud函数，P(s)是在句子空间定义的均匀分布。

where vs and vs′ are two different representa- tions of sentence s via two differently parameter- ized BERT encoders;， σ is the sigmoid function，and P (s) is a uniform distribution defined over the sentence space.

为了实现以上假设，作者为每个正样本，采取五个负样本。

**相似矩阵：**

一旦获得文档D的句子表示，作者采用成对求点积的方式来得到一个未标准化的相似矩阵：
$$
E^ ̄_{ij} =v_i^⊤v_j
$$
标准化：
$$
E^ ̃_{ij} = E^ ̄_{ij}−[minE^ ̄ +β(maxE^ ̄ − minE^ ̄)]
$$

$$
E_{ij}= E^ ̃_{ij}, \ if \ E^ ̃>0 \ , \ else \ 0
$$



Equation (5) aims to remove the effect of absolute values by emphasizing the relative contribution of different similarity scores. This is particularly im- portant for the adopted sentence representations which in some cases might assign very high values to all possible sentence pairs. Hyper-parameter β (β ∈ [0, 1]) controls the threshold below which the similarity score is set to 0.

### 4. **Evaluation**

数据集：NYT 与 CNN/Daily Mai，中文数据集TTNews

超参数：

- 优化器：Adam
- 初始学习率：4e-6

评价指标：

ROUGE-1、ROUGE-2、ROUGE-L

结果：

英文数据集，分别与SOTA有监督方法、无监督方法进行比较

![image-20211106162215196](https://tva1.sinaimg.cn/large/008i3skNgy1gw5i8qwpg0j31dg0oejx2.jpg)

超参数调优

![image-20211106162335953](https://tva1.sinaimg.cn/large/008i3skNgy1gw5ia3nrtmj30n60kwq4o.jpg)

中文数据集，与有监督、无监督方法进行对比的结果

![image-20211106162507741](https://tva1.sinaimg.cn/large/008i3skNgy1gw5ibp0dt1j30o60iiac8.jpg)

人工评测，通过构造QA，对生成的摘要内容进行人工评测的结果：

![image-20211106163903788](https://tva1.sinaimg.cn/large/008i3skNgy1gw5iq6w6noj30mq08k3zb.jpg)

### 5. **Conclusion**

作者提出的方法在三个数据集上都取得了比当前无监督文本抽取摘要基线模型更好的效果，并在英文数据集上取得了与有监督模型，如Pointer-generator相近的分数。

Experimental results on three news summarization datasets demonstrated the superiority of our approach against strong baselines.

未来，作者希望将本文提出的想法运用于有监督模型或者多文档摘要当中。

In the future, we would like to investigate whether some of the ideas introduced in this paper can improve the perfor- mance of supervised systems as well as sentence selection in multi-document summarization. 

### 6. **Notes**



### 7. **References**





