# LLM ROUTE

## Stage One

### Pytorch

- 入门知识
  看B站小土堆 或者 我记录的[笔记](https://chzarles.github.io/DeepLearning/Pytorch_tudui_intro)
- 进阶知识
  - [《PyTorch深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys/?spm_id_from=333.337.search-card.all.click&vd_source=27d3b33a76014ebb5a906ad40fa382de)
  - 学会看文档...
- https://www.huaxiaozhuan.com/

> 需要【达到的效果】
>
> - 会搭建深度学习环境和依赖包安装
> - 会使用常见操作，例如matmul, sigmoid, softmax, linear, relu等
> - dataset, dataloader, 损失函数，优化器的使用
> - 会用gpu手写训练和预测一个模型

### 学习 LLM 背景知识

- [入门书籍](https://intro-llm.github.io/chapter/LLM-TAP.pdf)

  - 重点章节（ch2，ch6

> 需要【达到的效果】
>
> - 了解预训练生成式模型和强化学习的原理

- 了解ChatGPT的发展脉络
  - 【OpenAI-大语言模型入门】https://www.bilibili.com/video/BV1Hj41177fb
  - 【论文研读】[InstructGPT: Training language models to follow instructions with human feedback](https://link.zhihu.com/?target=https://arxiv.org/pdf/2203.02155.pdf)
  - 【论文配套视频-李沐】https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=71b548de6de953e10b96b6547ada83f2

> 需要【达到的效果】
>
> 讲述出ChatGPT四阶段：预训练、有监督微调、奖励建模、强化学习 ，都分别做了什么

### 学习 Transformer/Bert/Gpt

这部分资料很多，尽量多看，每个材料讲的角度都不一样。学习的时候要重点关注, 位置编码, Self-Attention, 多头注意力机制（必学）, 预测解码（必学）

- 【图解+手撕底层原理】https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/篇章2-Transformer相关原理
- 【视频教程】https://www.bilibili.com/video/BV11v4y137sN
- 【动手写transformer】https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/篇章2-Transformer相关原理/2.2.1-Pytorch编写Transformer.ipynb
- 【Transformer实战项目】：https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/篇章4-使用Transformers解决NLP任务
- 【Attention 讲解】 https://www.zywvvd.com/notes/study/deep-learning/transformer/transformer-intr/transformer-intr-1/
- 【向量嵌入】：https://www.elastic.co/cn/what-is/vector-embedding
- 【Transformer系列】https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
- 【比较体系的文档 How-to-use-Transformers 】https://github.com/jsksxs360/How-to-use-Transformers

> 需要【达到的效果】
>
> - 手撕Transformer，能头到尾全部推导一遍
> - 手写核心模块Self-Attention的代码，最后达到不看公式也能写出来
> - 完成Transformer项目的数据处理和加载，预训练模型微调，模型预测，指标计算的完整项目流程，并总结实验报告。
