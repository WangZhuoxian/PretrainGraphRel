# PretrainGraphRel
一个使用预训练语言模型和图卷积网络进行关系分类的模型

## 依赖库
- Pytorch 1.3.1
- SpaCy 2.2.4
- Transformers 2.2.1

## 模型简介
用XLNet作为Encoder（因为有hugging face的[transformers](https://github.com/huggingface/transformers)库，所以也可以很方便地改用其他预训练模型），采用了和ACL2019的论文[Matching the Blanks Distributional Similarity for Relation Learning](https://www.aclweb.org/anthology/P19-1279.pdf)同样的实体位置编码方法，
在Encoder之上额外增加一个GCN层提取结构特征，结合XLNet提取的特征进行预测。

## 数据
- KBP37
- SemEval 2010 Task 8

