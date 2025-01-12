## 项目信息

https://cs50.harvard.edu/ai/2024/projects/6/attention/

## 项目背景

创建语言模型的一种方法是构建掩码语言模型（Masked Language Model），在该模型中，语言模型被训练以预测从文本序列中缺失的“掩码”单词。
BERT 是由谷歌开发的基于 transformer 的语言模型，其训练方法就是通过这种方式：语言模型根据周围的上下文单词来预测一个被掩码的单词。

BERT 使用transformer架构，因此采用attention机制来理解语言。在基础 BERT 模型中，transformer使用 12 层，每层有 12 个自注意力头，总共有 144 个自注意力头。

该项目将涉及两个部分：

    1.首先，我们将使用由 AI 软件公司 Hugging Face 开发的 transformers Python 库，编写一个程序，利用 BERT 预测被掩码的单词。该程序还将生成可视化注意力分数的图表，为每个自注意力头生成一个图表，共 144 个。
    2.其次，我们将分析程序生成的图表，以尝试理解 BERT 的注意力头在试图理解自然语言时可能关注的内容。

## 项目介绍

首先，查看 `mask.py` 程序。在主函数中，用户首先被提示输入一些文本。
该文本输入应包含一个掩码标记 [MASK]，表示我们的语言模型应该尝试预测的单词。
然后，该函数使用 AutoTokenizer 将输入拆分为一系列token。

在 BERT 模型中，每个不同的token都有其自己的 ID 号。tokenizer.mask_token_id 对应于 [MASK] 这个掩码标记。
大多数其他token代表单词，但也有一些例外。[CLS] token 总是出现在文本序列的开头。[SEP] token 出现在文本序列的末尾，用于将序列彼此分开。
有时，一个单词会被拆分成多个token：例如，BERT 将单词 “intelligently” 视为两个token：intelligent 和 ##ly。

接下来，我们用这个实现了BERT模型的api ( [TFBertForMaskedLM](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.TFBertForMaskedLM) )，
来预测 masked token。

在获得推理结果后，我们取出其中 top K tokens。并依次替换[MASK],然后打印输出。

最后，程序调用 visualize_attentions 函数，该函数应生成每个 BERT 注意力头对输入序列的注意力值的图示。

## 项目详细说明

需要实现的接口有 get_mask_token_index、get_color_for_attention_score 和 visualize_attentions 。

### get_mask_token_index

```python

def get_mask_token_index(mask_token_id:int, inputs:transformers.BatchEncoding
) -> Union[None, int]:
    """
    parameter:
        mask_token_id: [MASK]掩码对应的token id
        inputs: transformers.BatchEncoding
    return:
        The index of the mask token in the input ids, or None if not found. (0-indexd)
    """

```

### get_color_for_attention_score

get_color_for_attention_score 函数应接受一个注意力得分（一个介于 0 和 1 之间的值，包括 0 和 1），并输出一个包含三个整数的元组，代表用于该注意力单元在注意力图中的颜色的 RGB 三元组。

如果注意力得分为 0，颜色应为完全黑色（值为 (0, 0, 0)）。如果注意力得分为 1，颜色应为完全白色（值为 (255, 255, 255)）。对于介于两者之间的注意力得分，颜色应为与注意力得分线性缩放的灰色阴影。

要使颜色成为灰色阴影，红色、蓝色和绿色的值应相等。

红色、绿色和蓝色的值必须都是整数，但您可以选择截断或四舍五入这些值。例如，对于注意力得分 0.25，您的函数可以返回 (63, 63, 63) 或 (64, 64, 64)，因为 255 的 25% 是 63.75。

### visualize_attentions

visualize_attentions 函数接受一个token序列（字符串列表）以及 attentions，attentions 是一个 包含了多个tensor的tuple， tensor里保存的是注意力得分。
对于每个注意力头，该函数应生成一个注意力可视化图，通过调用 generate_diagram 来实现。

> generate_diagram 函数期望前两个输入为层号和头号。这些数字应为 1-indexed 索引。换句话说，对于第一个注意力头和注意力层（每个的索引为 0），layer_number 应为 1，head_number 也应为 1。

要索引 attentions 值以获得特定注意力头的值，你可以用 attentions\[i\]\[j\]\[k\] 来做，其中 i 是注意力层的索引，j 是beam的索引（在我们的例子中始终为 0），k 是层中注意力头的索引。

> 我也不懂beam是什么， [基于Attention的Seq2Seq与Beam搜索](https://www.lelovepan.cn/2018/08/07/%E5%9F%BA%E4%BA%8EAttention%E7%9A%84Seq2Seq%E4%B8%8Ebeam%E6%90%9C%E7%B4%A2.html)

该函数包含一个现有的实现，仅生成第一个注意力层中第一个注意力头的单个注意力图。您的任务是扩展此实现，以生成所有注意力头和层的图。

## Hints

在分析注意力图时，你经常会发现许多注意力头中的许多标记都强烈关注 [SEP] 或 [CLS] 标记。如果给定的注意力头中没有值得注意的词，就会发生这种情况。
