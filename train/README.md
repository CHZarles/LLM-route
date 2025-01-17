# intro

-
- 深度学习 - 第7篇 - 国产Ai芯片CUDA生态兼容的一些思考

【底层原理】分布式并行技术全家桶
● 【入门书籍】https://intro-llm.github.io/ （第4章）
● 【视频教程】https://www.bilibili.com/video/BV1ve411w7DL
● 【重要论文】
○ https://www.vldb.org/pvldb/vol13/p3005-li.pdf(数据并行，pytorch官方库DDP实现)
○ https://engineering.fb.com/2021/07/15/open-source/fsdp/(数据+模型并行，pytorch官方库FSDP实现)
○ https://arxiv.org/pdf/1811.06965.pdfarxiv.org(流水线并行, 谷歌GPipe)
○ https://arxiv.org/pdf/1909.08053.pdf(张量并行，微软Megatron-LM)
○ https://arxiv.org/pdf/2006.16668.pdf(MOE并行，谷歌GShard)
○ https://arxiv.org/pdf/1910.02054.pdf(多维混合并行，微软ZeRO)
○ https://arxiv.org/pdf/2105.13120.pdf(序列并行，Colossal-AI )
○ https://arxiv.org/pdf/1811.02084.pdf(自动并行， Mesh-Tensorflow)
○ https://arxiv.org/pdf/2104.04473.pdf(NVIDIA综述，数据/模型/张量/流水线并行)
● 【代码实战1】https://pytorch.apachecn.org/2.0/tutorials/intermediate/FSDP_tutorial/（数据+模型并行，Pytorch FSDP实战）
● 【代码实战2】https://h-huang.github.io/tutorials/intermediate/pipeline_tutorial.html（流水线并行）
● 【代码实战3】https://oslo.eleuther.ai/TUTORIALS/tensor_model_parallelism.html（1D/2D/2.5D/3D张量并行，OLSO框架实战）
● 【怎么学】这一阶段会安排经典论文的阅读。本阶段主要是LLM分布式训练理论和底层原理的学习，会配合一些小代码实战。这块市面上大模型训练框架所集成的技术已经比较成熟了，没有必要全部从头实现一遍，主要是应对面试的深度考察。先跟随入门书籍把第4章完整看完，了解分布式训练技术的基本原理。学习过程中可以配合视频教程一起看，效果会更好。这一阶段安排了经典论文阅读的任务，把列出来的9篇重要的论文精读一遍，看不懂的可以网上搜索对应的论文讲解，看完要做论文笔记。最后是完成代码实战，及pytorch官方的FSDP数据+模型并行的代码实现。这里最好找2张以上的GPU卡，可以无缝模拟真实的分布式训练的场景。
● 【学习的重点】
○ 数据并行
○ 模型并行
○ 流水线并行
○ 张量并行
● 【达到的效果】重点的4种并行技术，数据/模型/流水线/张量并行，可以完成手画原理图+自由讲解的程度。每种技术的优缺点，后续的改进优化方向也需要重点了解。
