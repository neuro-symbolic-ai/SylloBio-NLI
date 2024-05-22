import torch.cuda
from typing import List, Union
from langchain_core.prompts import PromptTemplate
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCT_MODELS = ["google/gemma-7b-it", "NousResearch/Hermes-2-Pro-Llama-3-8B"]

class LocalLLM:
    def __init__(self, model_locator: str):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_locator = model_locator
        self.tokenizer = AutoTokenizer.from_pretrained(model_locator, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_locator, device_map="auto", torch_dtype="auto",
                                                          offload_buffers=True)

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.llm = HuggingFacePipeline.from_model_id(
        #     model_id=model_locator,
        #     task="text-generation",
        #     device=1,
        #     device_map="auto",
        #     pipeline_kwargs={"max_new_tokens": output_size},
        # )

    def prompt(self, template: str, question: Union[str, List[str]], output_size: int) -> List[str]:
        prompt = PromptTemplate.from_template(template)
        # chain = prompt | self.llm
        #
        # return chain.invoke({"question": question}).removeprefix(prompt.format(question=question)).strip()

        prompts = list()
        if (isinstance(question, list)):
            prompts = [prompt.format(question=q) for q in question]
        else:
            prompts = [prompt.format(question=question)]

        if ("instruct" in self.model_locator.lower() or self.model_locator.lower() in INSTRUCT_MODELS):
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            inputs = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            tokens = self.tokenizer(inputs, padding=True, return_tensors="pt")
        else:
            tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")

        generated = self.model.generate(tokens.input_ids.to(self.model.device),
                                        attention_mask=tokens.attention_mask.to(self.model.device),
                                        max_new_tokens=output_size)

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i in range(len(decoded)):
            decoded[i] = decoded[i].replace(prompts[i], "").replace(".", "").strip()

        return decoded

