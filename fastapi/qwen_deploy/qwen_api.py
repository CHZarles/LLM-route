import asyncio
from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)


class Qwen:
    def __init__(self, model_path="./Qwen/Qwen1.5-7B-Chat", device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    async def stream_chat(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=2048)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
        thread.join()

    def generate_response(self, messages, stream=False):
        # inputs = self.build_inputs(query, history=history)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_token_count = len(model_inputs["input_ids"][0])
        if stream:
            raise NotImplementedError("streaming is not implemented")
        else:
            generated_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_token_count = len(generated_ids[0])
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return {
            "prompt_tokens": input_token_count,
            "completion_tokens": output_token_count,
            "generated_text": response,
        }


if __name__ == "__main__":
    qwen = Qwen()
    prompt = input("input your prompt: ")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = qwen.generate_response(messages)
    print("response: ", response)

    # test stream_chat, the ans will be output to terminal
    print(" =======================> test stream_chat <========================== ")

    async def test_stream():
        async for res in qwen.stream_chat(messages):
            print(res)

    asyncio.run(test_stream())
