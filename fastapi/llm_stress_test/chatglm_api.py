import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_prompt(query, history=None):
    if history is None:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
            i + 1, old_query, response
        )
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    return prompt


class ChatGLMModel:
    def __init__(self, model_path="./models/chatglm-6b-int4"):
        # check whether CUDA is available
        if torch.cuda.is_available():
            # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = None
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        ).half()
        model.to(self.torch_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        return model, tokenizer

    def build_inputs(self, query: str, history: List[Tuple[str, str]] = None):
        prompt = build_prompt(query, history=history)
        print("prompt: ", prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.torch_device)
        return inputs

    def generate_response(
        self, query: str, history: List[Tuple[str, str]], parameters=None
    ):
        if parameters is None:
            parameters = (True, 2048, 1)

        do_sample, max_length, num_beams = parameters
        set_random_seed(42)
        # inputs = self.tokenizer(query, return_tensors="pt")
        # inputs = inputs.to(self.torch_device)
        # print(query, history)
        inputs = self.build_inputs(query, history)
        input_token_count = len(inputs["input_ids"][0])
        outputs = self.model.generate(
            **inputs, do_sample=do_sample, max_length=max_length, num_beams=num_beams
        )
        output_token_count = len(outputs[0])
        outputs = outputs.tolist()[0]
        out_sentence = self.tokenizer.decode(outputs, skip_special_tokens=True)

        return {
            "prompt_tokens": input_token_count,
            "completion_tokens": output_token_count,
            "generated_text": out_sentence,
        }


if __name__ == "__main__":
    query = "他的出生时间是什么时候？"
    chat_glm_model = ChatGLMModel()
    result = chat_glm_model.generate_response(query, [("一人之下的作者是谁", "鸟山明")])
    print(result)
