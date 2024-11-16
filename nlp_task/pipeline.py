import json

import datasets

from bleu_metrics import BLEU

model_checkpoint = "uer/t5-base-chinese-cluecorpussmall"


# 1. 加载数据集
train_path = "./DuReaderQG/train.json"
dev_path = "./DuReaderQG/dev.json"
raw_datasets = datasets.load_dataset(
    "text", data_files={"train": train_path, "dev": dev_path}
)

from functools import partial

# 2. 数据集预处理
## 2.1 写一个map，将数据集中的每个样本转换成模型输入格式
from transformers import AutoTokenizer

from utils import convert_example

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
preprocess_function = partial(
    convert_example,
    tokenizer=tokenizer,
    max_source_seq_len=256,
    max_target_seq_len=32,
)
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


import torch

# 3. 创建模型
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4. 设置评估函数，评估函数用 BLUE1 - BLUE4 来评估模型的效果
import numpy as np

# 参考 https://github.com/zyds/transformers-code/blob/master/02-NLP%20Tasks/15-text_summarization/summarization.ipynb

bleu_evaluators = [BLEU(n_size=i + 1) for i in range(4)]


def compute_metrics(pred):
    print("call metrics")
    predictions, labels = pred
    for prediction, label in zip(predictions, labels):
        # print(prediction.shape)
        # print(label.shape)
        for bleu_evaluator in bleu_evaluators:
            bleu_evaluator.add_instance(prediction=prediction, references=[label])
    result = {
        "bleu1": bleu_evaluators[0].compute(),
        "bleu2": bleu_evaluators[1].compute(),
        "bleu3": bleu_evaluators[2].compute(),
        "bleu4": bleu_evaluators[3].compute(),
    }
    return result


# 5. 设置训练参数
from transformers import Seq2SeqTrainingArguments

batch_size = 16
train_args = Seq2SeqTrainingArguments(
    "QuestionAnswer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=30,
    predict_with_generate=True,
    fp16=True,
)


# 6. 创建训练器 并 训练
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# 7. 运行模型

from transformers import pipeline

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

question = "治疗宫颈糜烂的最佳时间"
context = "专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。"
input_str = "问题：{question}{sep_token}原文：{context}".format(
    question=question, context=context, sep_token=tokenizer.sep_token
)
print(input_str)
print(pipe(input_str))
