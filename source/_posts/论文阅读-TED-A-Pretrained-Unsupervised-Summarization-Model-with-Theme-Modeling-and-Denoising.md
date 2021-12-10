---
title: >-
  论文阅读-TED: A Pretrained Unsupervised Summarization Model with Theme Modeling
  and Denoising
date: 2021-11-06 17:01:59
tags: [论文]
mathjax: true
---

## TED: A Pretrained Unsupervised Summarization Model with Theme Modeling and Denoising

*一种基于主题建模和去噪的无监督预训练摘要生成模型*

**{2020.EMNLP. Ziyi Yang. Chenguang Zhu}**

```
@misc{yang2020ted,
      title={TED: A Pretrained Unsupervised Summarization Model with Theme Modeling and Denoising}, 
      author={Ziyi Yang and Chenguang Zhu and Robert Gmyr and Michael Zeng and Xuedong Huang and Eric Darve},
      year={2020},
      eprint={2001.00725},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### **Summary**



### **Research Objective(s)**

1. 以往的模型基于RNN实现，而新提出的transformer性能更好
2. 以往的摘要模型忽略了在大规模未标注语料的预训练

基于上述问题，作者提出基于Transformer的无监督生成模型。模型基于以下思路：

- 首先在未标注的大规模语料上进行训练
- 对TED模型基于主题模型与去噪自编码器提高生成摘要的质量

### **Background / Problem Statement**

作者提出一种基于Transformer的无监督生成模型，贡献体现在：

1. TED预训练：

   **新闻文体采用倒金字塔结构，可用新闻的开头若干句，作为摘要内容**。基于此假设，作者对模型在大规模未标注的语料上进行无监督预训练

2. TED finetune：

   在具体的数据集上进行finetune，作者基于**主题模型损失**以及**去噪自编码器**进行finetune。

   其中，主题模型loss目标是生成的文本与原始文本**语义上更相近**，而去噪自编码器目标是帮助模型从原始文本中，**抽取主要信息**

3. 为了解决生成模型常见的OOV问题，作者采用了**SentencePiece tokenization**^[1]^

### **Method(s)**

1. 基于Transformer的Encoder-Decoder结构

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gx4d4pyey1j31fo0hodib.jpg" alt="image-20211206200009560" style="zoom:50%;" />

2. 无监督预训练
3. 主题模型
4. 去噪自编码器

### **Evaluation**



### **Conclusion**



### **Notes**



### **References**

[1] Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tok- enizer and detokenizer for neural text processing. *arXiv preprint arXiv:1808.06226*.

