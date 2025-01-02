from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    def __init__(self, model_path="./Qwen/Qwen1.5-7B-Chat", device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_response(self, messages):
        # inputs = self.build_inputs(query, history=history)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_token_count = len(model_inputs.input_ids)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
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
