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
    max_source_seq_len=512,
    max_target_seq_len=512,
)
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


# 3. 创建模型
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 4. 设置评估函数，评估函数用 BLUE1 - BLUE4 来评估模型的效果
import numpy as np

# 参考 https://github.com/zyds/transformers-code/blob/master/02-NLP%20Tasks/15-text_summarization/summarization.ipynb

bleu_evaluators = [BLEU(n_size=i+1) for i in range(4)]

def compute_metrics(pred):
    print("call metrics")
    predictions, labels = pred
    for bleu_evaluator in bleu_evaluators:
        bleu_evaluator.add_instance(prediction=predictions, references=[babels])
    return [bleu.compute() for bleu in bleu_evaluators]


# 5. 设置训练参数
from transformers import Seq2SeqTrainingArguments

batch_size = 4
train_args = Seq2SeqTrainingArguments(
    "QuestionAnswer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
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


# 7. 保存模型

from transformers import pipeline

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

context = "选择燃气热水器时，一定要关注这几个问题：1、出水稳定性要好，不能出现忽热忽冷的现象2、快速到达设定的需求水温3、操作要智能、方便4、安全性要好，要装有安全报警装置 市场上燃气热水器品牌众多，购买时还需多加对比和仔细鉴别。方太今年主打的磁化恒温热水器在使用体验方面做了全面升级：9秒速热，可快速进入洗浴模式；水温持久稳定，不会出现忽热忽冷的现象，并通过水量伺服技术将出水温度精确控制在±0.5℃，可满足家里宝贝敏感肌肤洗护需求；配备CO和CH4双气体报警装置更安全（市场上一般多为CO单气体报警）。另外，这款热水器还有智能WIFI互联功能，只需下载个手机APP即可用手机远程操作热水器，实现精准调节水温，满足家人多样化的洗浴需求。当然方太的磁化恒温系列主要的是增加磁化功能，可以有效吸附水中的铁锈、铁屑等微小杂质，防止细菌滋生，使沐浴水质更洁净，长期使用磁化水沐浴更利于身体健康。"
question = "燃气热水器哪个牌子好"
input_str = "问题：{question}{sep_token}原文：{context}".format(
    question=question, context=context, sep_token=tokenizer.sep_token
)
print(input_str)
print(pipe(input_str))
