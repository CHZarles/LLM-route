import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: initialize the model and the tokenizer
model_name = "gpt2"
tokennizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
"""
# NOTE:  tokenizing the prompt
prompt = "The quick brown fox jumps over the"
inputs = tokennizer(prompt, return_tensors="pt")
print(inputs)

"""
{'input_ids': tensor([[  464,  2068,  7586, 21831, 18045,   625,   262]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

input_ids : the mapping from particular chunks of the text 
            into these tokens from the tokniizer

attention_mask: a tensor that goes along with input_ids for
                the purpose of determining what tokens within
                the particular input_ids are should be attended to
                which other tokens. (there is 1 for each input token id.)
"""


# NOTE: predict and look at the shape of the logits tensor
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print(logits.shape)
"""
torch.Size([1, 7, 50257])

1 - batch size
7 - the sequence length that we pass in
50257 - the vocabulary size which how many possible tokens that we might 
        generate in the output

"""

# NOTE: to get the logits for the last token in the input sequence
last_logits = outputs.logits[0, -1, :]
"""
[0, -1, :] 

"0" : the first batch
"-1": correspond to the last token in the input sequence
      because we are trying to predict the next token of the input sequence
      so we just focus on the last token 

":": all the possible tokens that we might generate in the output

"""

# NOTE : to get the most likely token that we might generate in the output
next_token_id = last_logits.argmax()
print(next_token_id)
"""
the most likely token that we might generate in the output
tensor(13990)
"""

next_token = tokennizer.decode(next_token_id)
print(next_token)
"""
fence
"""

# NOTE: to get the top 10 most likely tokens that we might generate in the output
top_k = torch.topk(last_logits, k=10)
tokens = [tokennizer.decode(token_id) for token_id in top_k.indices]
print(tokens)
"""
the top 10 most likely tokens that we might generate in the output

how do we use these probabilities to generate text?
This is where decoding strategies, such as greedy search and beam search, 
come into play.


ref :
[Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
"""

# NOTE: how to construct the next input ?
next_inpusts = {
    "input_ids": torch.cat([inputs["input_ids"], next_token_id.reshape((1, 1))], dim=1),
    "attention_mask": torch.cat(
        [
            inputs["attention_mask"],
            torch.ones((1, 1)).type_as(inputs["attention_mask"]),
        ],
        dim=1,
    ),
}


# NOTE: Furthemore, let's try to calculate the time taken to generate the next token
def generate_token(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id


generated_tokens = []
next_inputs = inputs
durations_s = []
for _ in range(20):
    t0 = time.time()
    next_token_id = generate_token(next_inputs)
    durations_s += [time.time() - t0]

    next_inputs = {
        "input_ids": torch.cat(
            [next_inputs["input_ids"], next_token_id.reshape((1, 1))], dim=1
        ),
        "attention_mask": torch.cat(
            [
                next_inputs["attention_mask"],
                torch.tensor([[1]]),
            ],
            dim=1,
        ),
    }

    next_token = tokennizer.decode(next_token_id)
    generated_tokens.append(next_token)

print(f"sum(durations_s) = {sum(durations_s)} s")
print(f"generated_tokens = {generated_tokens}")


# NOTE: try to look at the time consumed for each token generation
import matplotlib.pyplot as plt

plt.plot(durations_s)
plt.show()
