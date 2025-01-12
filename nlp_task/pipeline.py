import json

import datasets
from torch.utils.tensorboard import SummaryWriter

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


writer = SummaryWriter("training_logs")


train_dataset_size = len(tokenized_datasets["train"])
batch_size = 12
step_each_epoch = train_dataset_size // batch_size
echo_times = 0


# tensorbard 显示问题
# https://github.com/tensorflow/tensorboard/issues/6808
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
    global echo_times
    global step_each_epoch
    echo_times += 1
    print("echo_times * step_each_epoch", echo_times * step_each_epoch)
    for i in range(4):
        writer.add_scalar(
            "eval/bleu-size-{}".format(i + 1),
            result[f"bleu{i + 1}"],
            echo_times * step_each_epoch,
        )
    return result


# 5. 设置训练参数
from transformers import Seq2SeqTrainingArguments

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

model.save_pretrained("./CharlesQuestinAnswer")

# 7. 运行模型

from transformers import pipeline

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

question = "广州英语培训哪家好"
context = "广州学英语,我认为一定要找对学校,否则花了钱还浪费了时间。广州的英语培训学校,大多是夜校性质,走读那种,通常是来也匆匆,去也匆匆,没有多少停留操练或静心读书的机会。遇到滂沱大雨或交通拥堵,缺课旷课是常有的事。 鉴于上述情况,我建议你报读全封闭式英语培训学校。广州有一家是专门做封闭式英语口语强化训练的学校,叫广州东方英文书院。这家培训机构从事全封闭英语教学17年,很少卖广告,从来就不搞什么招生噱头,吸引眼球;而是注重口碑宣传,因为口碑是最好的广告;而口碑是靠质量去打造。该书院位于山清水秀的五A级白云山里面,环境一流,中国唯一一家能将培训学校办在五A级风景区里面的培训学校。我曾经有同事去读过,觉得学英语确实需要一个安静的环境,这样才能做到六根清净潜心读书,否则匆匆而来、急急而去,天天走读,很难有一个很好的连贯性,进步显然没有全封闭式培训那么明显。要在广州学英语,不如索性封闭起来,将空余时间用到尽,也许能让你学有所成。"
input_str = "问题：{question}{sep_token}原文：{context}".format(
    question=question, context=context, sep_token=tokenizer.sep_token
)
# print(input_str)
print(pipe(input_str))
