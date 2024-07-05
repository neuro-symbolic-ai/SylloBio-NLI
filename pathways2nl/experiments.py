import json
import os.path
import random
from typing import Tuple, List, Dict
from itertools import chain
from dynaconf import settings
from tqdm import tqdm
from .pathways import (
    PathwaySetOps, get_ph_tuples, falsify, add_distractors_all, SyllogisticScheme, SyllogisticSchemeVariant
)
from .llm import LocalLLM


class SyllogisticReasoningTest:
    model_cache = {"locator": None, "llm": None}

    def __init__(self, model_locator: str, scheme: SyllogisticScheme, variant: SyllogisticSchemeVariant, dummy: bool,
                 num_premises: int = 2, icl: bool = False, seed: int = 0, batch_size: int = 10, subset_size: int = 200):
        with open(settings["prompt_template_file_path"]) as templ_file:
            self.templates = json.load(templ_file)
        self.model_locator = model_locator
        self.scheme = scheme
        self.variant = variant
        self.dummy = dummy
        self.icl = icl
        self.num_premises = num_premises
        self.seed = seed
        self.batch_size = batch_size

        self._pwops = PathwaySetOps()

        ds_fname = f"datasets/{self.conf_string('', 0, False)}_dataset.json"
        if (os.path.exists(ds_fname)):
            with open(ds_fname) as ds_file:
                self._ph_tuples = json.load(ds_file)
        else:
            self._ph_tuples = get_ph_tuples(self._pwops, scheme, variant, num_premises, dummy, subset_size)
            os.makedirs("datasets", exist_ok=True)
            with open(ds_fname, "w") as ds_file:
                json.dump(self._ph_tuples, ds_file, indent=2)

        if (model_locator == SyllogisticReasoningTest.model_cache["locator"]):
            self._llm = SyllogisticReasoningTest.model_cache["llm"]
        else:
            del SyllogisticReasoningTest.model_cache["llm"]
            self._llm = LocalLLM(model_locator)
            SyllogisticReasoningTest.model_cache["locator"] = model_locator
            SyllogisticReasoningTest.model_cache["llm"] = self._llm

        self.task_map = {"TASK_1": self.task_1, "TASK_2": self.task_2}

    @staticmethod
    def calc_metrics_binary(results: Dict[str, List[bool]]):
        metrics = {
            "precision": sum(results["True"]) / len(results["True"]),
            "recall": sum(results["True"]) / (sum(results["True"]) + sum([not (r) for r in results["False"]]))
                      if sum(results["True"]) else 0.0,
            "accuracy": sum(results["True"] + results["False"]) / (len(results["True"]) * 2)
        }

        if (metrics["precision"] < 1e-6):
            metrics["f1-score"] = 0.0
        else:
            metrics["f1-score"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

        return metrics

    @staticmethod
    def calc_metrics_premises(results: Dict[str, List[bool]], num_premises: int, num_distractors: int):
        distraction_total = sum(chain(*[results[f"P{i + 1}"] for i in range(num_premises, num_premises + num_distractors)]))
        joint_results = [sum(pj) == len(pj) for pj in zip(*[results[f"P{i + 1}"] for i in range(num_premises)])]
        metrics = {
            "p_accuracy": {f"P{i + 1}": sum(results[f"P{i + 1}"]) / len(results[f"P{i + 1}"]) for i in range(num_premises)},
            "joint_accuracy": sum(joint_results) / len(joint_results),
            "distraction_rate": distraction_total / (len(results["P1"]) * num_distractors) if (num_distractors > 0) else 0.0
        }

        return metrics

    def conf_string(self, task: str, num_distractors: int, icl: bool) -> str:
        conf_str = f"{task.lower()}-" if task else ""
        conf_str += f"{self.scheme}-{self.num_premises}-{num_distractors}-{self.variant}"
        conf_str += f"-{'dummy' if self.dummy else 'real'}{'-icl' if icl else ''}"

        return conf_str


    def load_logged(self) -> Tuple[List[str], List[str]]:
        answers = list()
        gtd = list()
        if (self._llm.logging_conf):
            task = self._llm.logging_conf['task']
            scheme = self._llm.logging_conf['scheme']
            variant = self._llm.logging_conf['variant']
            num_prem = self._llm.logging_conf['num_prem']
            num_distr = self._llm.logging_conf['num_distr']
            dummy = self._llm.logging_conf['dummy']
            icl = self._llm.logging_conf['icl']
            conf_string = f"{task}-{scheme}-{num_prem}-{num_distr}-{variant}-{'dummy' if dummy else 'real'}{'-icl' if icl else ''}"
            log_fname = f"logs/{self.model_locator.replace('/', '--')}_{conf_string}_prompts_log.jsonl"

            if (os.path.exists(log_fname)):
                with open(log_fname) as log_file:
                    for line in log_file:
                        plog = json.loads(line)
                        answers.append(plog["cleaned"])
                        gtd.append(plog["gtd"])

        return answers, gtd

    def task_1(self, ph_tuples: List[Dict[str, str]], num_distractors: int):
        answers, gtd = self.load_logged()

        if (not answers):
            tuple_sets = {"True": ph_tuples, "False": falsify(ph_tuples)}
            results = {truth_val: list() for truth_val in tuple_sets}
            questions = list()
            for truth_val in tuple_sets:
                for ph_tuple in tuple_sets[truth_val]:
                    question = "\n".join([f"{key}: {stt}" for key, stt in ph_tuple.items()])
                    questions.append((question, truth_val))

            icl_examples = "Demonstration:\n\n" + '\n\n'.join(self.templates['EXAMPLES_TASK1']) + "\n\nTo determine:\n\n"

            batch_size = self.batch_size
            for i in tqdm(range(len(questions) // batch_size + int(len(questions) % batch_size > 0)),
                          desc=f"Prompting [{self.model_locator}] for {self.scheme} ({self.variant}) with {num_distractors} distractors"):
                answers = self._llm.prompt(self.templates["CONTEXT"] + self.templates["TASK_1"],
                                           [q[0] for q in questions[i * batch_size: i * batch_size + batch_size]],
                                           icl_examples if self.icl else "",
                                           [q[1] for q in questions[i * batch_size: i * batch_size + batch_size]],
                                           (5 + 5 * self.num_premises))

                for ans, truth_val in zip(answers, [q[1] for q in questions[i * batch_size: i * batch_size + batch_size]]):
                    results[truth_val].append(ans.startswith(truth_val))
        else:
            print(f"Loaded prompts on [{self.model_locator}] for {self.scheme} with {num_distractors} distractors")
            results = {truth_val: list() for truth_val in ("True", "False")}
            for ans, truth_val in zip(answers, gtd):
                results[truth_val].append(ans.startswith(truth_val))

        return SyllogisticReasoningTest.calc_metrics_binary(results)

    def task_2(self, ph_tuples: List[Dict[str, str]], num_distractors: int):
        answers, gtd = self.load_logged()

        if (not answers):
            tuple_sets = {"True": ph_tuples, "False": falsify(ph_tuples)}
            truth_results = {truth_val: list() for truth_val in tuple_sets}
            premise_results = {f"P{i + 1}": list() for i in range(self.num_premises + num_distractors)}
            questions = list()
            for truth_val in tuple_sets:
                for ph_tuple in tuple_sets[truth_val]:
                    question = "\n".join([f"{key}: {stt}" for key, stt in ph_tuple.items()])
                    questions.append((question, truth_val))

            icl_examples = "Demonstration:\n\n" + '\n\n'.join(self.templates['EXAMPLES_TASK2']) + "\n\nTo determine:\n\n"

            batch_size = self.batch_size
            for i in tqdm(range(len(questions) // batch_size + int(len(questions) % batch_size > 0)),
                          desc=f"Prompting [{self.model_locator}] for {self.scheme} ({self.variant}) with {num_distractors} distractors"):
                answers = self._llm.prompt(self.templates["CONTEXT"] + self.templates["TASK_2"],
                                           [q[0] for q in questions[i * batch_size: i * batch_size + batch_size]],
                                           icl_examples if self.icl else "",
                                           [q[1] for q in questions[i * batch_size: i * batch_size + batch_size]],
                                           (5 + 5 * self.num_premises))

                for ans, truth_val in zip(answers, [q[1] for q in questions[i * batch_size: i * batch_size + batch_size]]):
                    fields = ans.split(",")
                    truth_results[truth_val].append(fields[0].strip() == truth_val)
                    for i in range(self.num_premises + num_distractors):
                        premise_results[f"P{i + 1}"].append(bool(sum([field.strip().startswith(f"P{i + 1}") for field in fields])))
        else:
            print(f"Loaded prompts on [{self.model_locator}] for {self.scheme} with {num_distractors} distractors")
            truth_results = {truth_val: list() for truth_val in ("True", "False")}
            premise_results = {f"P{i + 1}": list() for i in range(self.num_premises + num_distractors)}
            for ans, truth_val in zip(answers, gtd):
                fields = ans.split(",")
                truth_results[truth_val].append(fields[0].strip() == truth_val)
                for i in range(self.num_premises + num_distractors):
                    premise_results[f"P{i + 1}"].append(
                        bool(sum([field.strip().startswith(f"P{i + 1}") for field in fields])))

        metrics = SyllogisticReasoningTest.calc_metrics_binary(truth_results)
        metrics |= SyllogisticReasoningTest.calc_metrics_premises(premise_results, self.num_premises, num_distractors)

        return metrics

    def run(self, task: str, num_distractors: int = 0, shuffle: bool = False) -> dict:
        ph_tuples = [pht.copy() for pht in self._ph_tuples]
        if (shuffle):
            random.seed(self.seed)
            random.shuffle(ph_tuples)

        self._llm.logging_conf = {
            "task": task.lower(),
            "scheme": self.scheme,
            "variant": self.variant,
            "num_prem": self.num_premises,
            "num_distr": num_distractors,
            "dummy": self.dummy,
            "icl": self.icl
        }

        if (num_distractors > 0):
            conf_string = self.conf_string("", num_distractors, False)
            ds_fname = f"datasets/{conf_string}_dataset.json"
            if (os.path.exists(ds_fname)):
                with open(ds_fname) as ds_file:
                    ph_tuples = json.load(ds_file)
            else:
                add_distractors_all(ph_tuples, self._pwops, n=num_distractors, seed=self.seed)
                with open(ds_fname, "w") as ds_file:
                    json.dump(ph_tuples, ds_file, indent=2)

        metrics = self.task_map[task](ph_tuples, num_distractors)

        return metrics






