import os
import json
import torch.cuda
from typing import List, Union
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCT_MODELS = ["google/gemma-7b-it", "NousResearch/Hermes-2-Pro-Llama-3-8B"]

class LocalLLM:
    def __init__(self, model_locator: str, logging_conf: dict = None):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_locator = model_locator
        self.tokenizer = AutoTokenizer.from_pretrained(model_locator, padding_side="left")
        if (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            self.model = AutoModelForCausalLM.from_pretrained(model_locator, device_map="auto", torch_dtype="auto",
                                                              attn_implementation="flash_attention_2",
                                                              offload_buffers=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_locator, torch_dtype="auto")

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logging_conf = logging_conf

    @staticmethod
    def clean_answers(answers: List[str]):
        answers = [ans.replace("*", "").replace(":", "") for ans in answers]
        answers = [ans.replace("user", "").replace("assistant", "").replace("model", "").strip() for ans in answers]
        answers = [ans.replace("#My", "").replace("The output is", "") for ans in answers]
        answers = [ans.replace("Answer", "").replace("The answer is", "") for ans in answers]
        answers = [ans.replace("[INST]", "").replace("[/INST]", "") for ans in answers]
        answers = [ans.replace("<", "").replace(">", "") for ans in answers]
        answers = [ans.strip() for ans in answers]

        return answers

    def log_prompt(self, prompts: List[str], decoded: List[str], clean_responses: List[str], gtd: List[str]):
        task = self.logging_conf['task']
        scheme = self.logging_conf['scheme']
        variant = self.logging_conf["variant"]
        num_prem = self.logging_conf['num_prem']
        num_distr = self.logging_conf['num_distr']
        dummy = self.logging_conf['dummy']
        icl = self.logging_conf['icl']
        conf_string = f"{task}-{scheme}-{num_prem}-{num_distr}-{variant}-{'dummy' if dummy else 'real'}{'-icl' if icl else ''}"
        log_fname = f"logs/{self.model_locator.replace('/', '--')}_{conf_string}_prompts_log.jsonl"

        os.makedirs("logs", exist_ok=True)
        with open(log_fname, "a") as log_file:
            for i in range(len(prompts)):
                log_file.write(
                    json.dumps({
                        "prompt": prompts[i],
                        "response": decoded[i],
                        "cleaned": clean_responses[i],
                        "gtd": gtd[i]
                    }) + "\n"
                )

    def prompt(self, template: str, question: Union[str, List[str]], examples: str, gtd: Union[str, List[str]],
               output_size: int) -> List[str]:
        prompt = PromptTemplate.from_template(template)

        prompts = list()
        if (isinstance(question, list)):
            prompts = [prompt.format(question=q, examples=examples) for q in question]
        else:
            prompts = [prompt.format(question=question, examples=examples)]

        if ("instruct" in self.model_locator.lower() or self.model_locator.lower() in INSTRUCT_MODELS):
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            inputs = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            tokens = self.tokenizer(inputs, padding=True, return_tensors="pt")
        else:
            tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")

        generated = self.model.generate(tokens.input_ids.to(self.model.device),
                                        attention_mask=tokens.attention_mask.to(self.model.device),
                                        max_new_tokens=output_size)

        decoded = list()
        for i in range(generated.shape[0]):
            response = generated[i][tokens.input_ids.shape[-1]:]
            decoded.append(self.tokenizer.decode(response, skip_special_tokens=True).replace(".", "").strip())

        clean_responses = LocalLLM.clean_answers(decoded)

        if (self.logging_conf):
            self.log_prompt(prompts, decoded, clean_responses, gtd)

        return clean_responses

