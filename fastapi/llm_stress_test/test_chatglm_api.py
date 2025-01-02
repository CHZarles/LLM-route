# ref
# https://huggingface.co/THUDM/chatglm-6b/blob/main/test_modeling_chatglm.py

import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# check whether CUDA is available
if torch.cuda.is_available():
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None


def set_random_seed(seed):

    random.seed(seed)

    # pytorch RNGs

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG

    np.random.seed(seed)


def get_model_and_tokenizer():
    model = AutoModel.from_pretrained(
        "./models/chatglm-6b-int4", trust_remote_code=True
    ).half()
    model.to(torch_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "./models/chatglm-6b-int4", trust_remote_code=True
    )
    return model, tokenizer


def test_generation(query: str):
    model, tokenizer = get_model_and_tokenizer()
    sentence = query
    parameters = [
        # (False, 2048, 1),
        # (False, 64, 1),
        (True, 2048, 1),
        # (True, 64, 1),
        # (True, 2048, 4),
    ]
    for do_sample, max_length, num_beams in parameters:
        set_random_seed(42)
        inputs = tokenizer(sentence, return_tensors="pt")
        inputs = inputs.to(torch_device)

        outputs = model.generate(
            **inputs, do_sample=do_sample, max_length=max_length, num_beams=num_beams
        )

        outputs = outputs.tolist()[0]
        out_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
        print(out_sentence)


if __name__ == "__main__":
    query = "What is the meaning of life?"
    test_generation(query)
