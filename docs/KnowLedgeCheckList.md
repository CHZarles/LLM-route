# 理论相关

## 1. 谈谈ChatGPT的优缺点

## 2. 请简述下Transformer基本流程

## 3. 为什么基于Transformer的架构需要多头注意力机制？

注意力机制

## 4. 编码器，解码器，编解码LLM模型之间的区别是什么？

## 5. 你能解释在语言模型中强化学习的概念吗？它如何应用于ChatGPT？

强化学习的是机器学习的一种方式，它的理念就是在模型训练的过程中，
通过不断给模型的输出来打分，来让模型学会做出高分的行为。

在语言模型的训练中，强化学习可以用来提高模型的生成答案的质量。

在ChatGpt的构建步骤里，研究人员会训练一个 Reward Modeling ，这个模型的作用是
对 SFT 模型给出的多个不同的输出结果的质量进行排序（对于同一个提示词）。

有了RM 就可以使用强化学习的方法，在 SFT 模型基础上调整参数，使得语言模型的表现更好。

> 参考：https://intro-llm.github.io/chapter/LLM-TAP.pdf 1.3节
> 强化学习和监督学习主要不同点在于：强化学习训练时，没有监督者给予指导，监督学习中训练数据存在明确的Label，学习过程中很清楚当前在学习哪类Label的特征。而强化学习需要通过和环境进行交互，环境给予反馈来推进学习过程。

## 6. 在GPT模型中，什么是温度系数？

## 7. 什么是旋转位置编码（ROPE）？

## 8. 为什么现在的大模型大多是decoder-only的架构？

## 9. ChatGPT的训练步骤有哪些？

## 10. 为什么transformers需要位置编码？

## 11. 为什么对于ChatGPT而言，提示工程很重要？

## 12. 如何缓解 LLMs 复读机问题？

# 部署相关

## 2.FastAPI如何处理GET请求的查询参数?

## 3.请说明一下FastAPI是如何支持异步IO的?

## 4.请阐述一下FastAPI是如何处理文件上传的?

## 5.async/await 原理是什么?

## 6.什么是协程?

协程是能暂停执行之后恢复的函数。python协程是无栈协程, 协程切换的上下文数据保存在堆中。
协程在暂停执行的时候,程序会跳回到caller,同时,caller的数据会被从堆恢复到栈.

> [【协程革命】理论篇！扫盲，辟谣一条龙！全语言通用，夯实基础，准备起飞！ 全程字幕](https://bilibili.com/video/BV1K14y1v7cw/?spm_id_from=333.999.0.0&vd_source=27d3b33a76014ebb5a906ad40fa382de)

## 7.chatglm3-6b有哪些低成本部署方式?

## 8.chatglm3-6b本地部署如何实现多个用户并发访问?

## 9.模型部署中FP32，FP16以及Int8的区别?

## 10.GPU显存占用和GPU利用率的定义

## 11.模型的FLOPs怎么算?

https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc

## 12.影响模型inference速度的因素?

## 13.阐述gevent的原理和工作机制
