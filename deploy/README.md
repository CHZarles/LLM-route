# 《Efficiently Serving LLMs》

source: https://www.bilibili.com/video/BV1bt421578n/?spm_id_from=333.337.search-card.all.click&vd_source=27d3b33a76014ebb5a906ad40fa382de

a video about key technical ideas of LLM model deployment.

## generate text one token at a time

In this lession , we will introduce the process of text generation,
using autoregressive language models, we will learn how to iteratively
generate text one token at a time from model outputs, and we will see
how this process can be divided into two phases, pre-fill and decode
, and optimize it using KV caching to speed up the attention computation.

## impl kvcache

## batch prompts into a single tensors

### continues batching

## quantization function

## Parameter effecient

### low rank adaptation

### multi lora

## conceptions

- autoregressive large language model
  > Autoregressive models are a type of statistical or machine learning model that predicts the next value in a sequence based on the previous values in that sequence.
  > These models assume that the future values in the sequence are dependent on the past values and use this dependency to make predictions.
  > --- [Autoregressive Models for Natural Language Processing](https://medium.com/@zaiinn440/autoregressive-models-for-natural-language-processing-b95e5f933e1f)
