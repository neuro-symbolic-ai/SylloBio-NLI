import json
import random
from typing import List, Dict
from itertools import chain
from dynaconf import settings
from tqdm import tqdm
from .pathways import load_hierarchy_genes, get_tree, get_ph_tuples, falsify, add_distractors
from .llm import LocalLLM


class SyllogisticReasoningTest:
    def __init__(self, model_locator: str, dummy: bool, num_premises: int = 2, seed: int = 0, batch_size: int = 10):
        with open(settings["template_file_path"]) as templ_file:
            self.templates = json.load(templ_file)
        self.model_locator = model_locator
        self.dummy = dummy
        self.num_premises = num_premises
        self.seed = seed
        self.batch_size = batch_size

        pw_hierarchy_genes = load_hierarchy_genes()
        self._pw_tree = get_tree(pw_hierarchy_genes)
        self._ph_tuples = get_ph_tuples(self._pw_tree, num_premises, include_genes=True, dummy=dummy)

        self._llm = LocalLLM(model_locator)

    @staticmethod
    def calc_metrics_binary(results: Dict[str, List[bool]]):
        metrics = {
            "precision": sum(results["True"]) / len(results["True"]),
            "recall": sum(results["True"]) / (sum(results["True"]) + sum([not (r) for r in results["False"]])),
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
            "distraction_rate": distraction_total / (len(results["P1"]) * num_distractors)
        }

        return metrics

    @staticmethod
    def clean_answers(answers: List[str]):
        answers = [ans.replace("*", "").replace(":", "") for ans in answers]
        answers = [ans.replace("user", "").replace("assistant", "").replace("model", "").strip() for ans in answers]
        answers = [ans.replace("#My", "").replace("The output is", "") for ans in answers]
        answers = [ans.replace("Answer", "").replace("The answer is", "") for ans in answers]
        answers = [ans.replace("[INST]", "").replace("[/INST]", "") for ans in answers]
        answers = [ans.strip() for ans in answers]

        return answers

    def task_1(self, ph_tuples: List[Dict[str, str]], subset_size: int, num_distractors: int):
        if (num_distractors > 0):
            add_distractors(ph_tuples, self._pw_tree, n=num_distractors, seed=self.seed)
        tuple_sets = {"True": ph_tuples[:subset_size], "False": falsify(ph_tuples[:subset_size])}
        results = {truth_val: list() for truth_val in tuple_sets}
        questions = list()
        for truth_val in tuple_sets:
            for ph_tuple in tuple_sets[truth_val]:
                question = "\n".join([f"{key}: {stt}" for key, stt in ph_tuple.items()])
                questions.append((question, truth_val))

        batch_size = self.batch_size
        for i in tqdm(range(len(questions) // batch_size + int(len(questions) % batch_size > 0)),
                      desc=f"Prompting [{self.model_locator}] with {num_distractors} distractors"):
            answers = self._llm.prompt(self.templates["TASK_1"],
                                       [q[0] for q in questions[i * batch_size: i * batch_size + batch_size]],
                                       (5 + 5 * self.num_premises))

            answers = SyllogisticReasoningTest.clean_answers(answers)

            with open(f"{self.model_locator.replace('/', '-')}_{num_distractors}_answers.txt", "a") as ans_file:
                ans_file.write("\n".join(answers))

            for ans, truth_val in zip(answers, [q[1] for q in questions[i * batch_size: i * batch_size + batch_size]]):
                results[truth_val].append(ans.startswith(truth_val))

        return SyllogisticReasoningTest.calc_metrics_binary(results)

    def task_2(self, ph_tuples: List[Dict[str, str]], subset_size: int, num_distractors: int):
        add_distractors(ph_tuples, self._pw_tree, n=num_distractors, seed=self.seed)
        tuple_sets = {"True": ph_tuples[:subset_size], "False": falsify(ph_tuples[:subset_size])}
        truth_results = {truth_val: list() for truth_val in tuple_sets}
        premise_results = {f"P{i + 1}": list() for i in range(self.num_premises + num_distractors)}
        questions = list()
        for truth_val in tuple_sets:
            for ph_tuple in tqdm(tuple_sets[truth_val], desc=f"Prompting [{truth_val}] with {num_distractors} distractors"):
                question = "\n".join([f"{key}: {stt}" for key, stt in ph_tuple.items()])
                answer = self._llm.prompt(self.templates["TASK_2"], question, (5 + 5 * self.num_premises))
                fields = answer.split(",")
                truth_results[truth_val].append(fields[0].strip() == truth_val)
                for i in range(self.num_premises + num_distractors):
                    premise_results[f"P{i + 1}"].append(f"P{i + 1}" in [field.strip() for field in fields])

        metrics = SyllogisticReasoningTest.calc_metrics_binary(truth_results)
        metrics |= SyllogisticReasoningTest.calc_metrics_premises(premise_results, self.num_premises, num_distractors)

        return metrics

    def run(self, task: str, subset_size: int, num_distractors: int = 0, shuffle: bool = False) -> dict:
        ph_tuples = [pht.copy() for pht in self._ph_tuples]
        if (shuffle):
            random.seed(self.seed)
            random.shuffle(ph_tuples)

        metrics = None
        if (task == "TASK_1"):
            metrics = self.task_1(ph_tuples, subset_size, num_distractors)
        elif (task == "TASK_2"):
            metrics = self.task_2(ph_tuples, subset_size, num_distractors)

        return metrics






