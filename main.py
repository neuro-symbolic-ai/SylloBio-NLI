import sys
import json
import pandas as pd
from pathways2nl.pathways import SyllogisticScheme, SyllogisticSchemeVariant
from pathways2nl.experiments import SyllogisticReasoningTest


def main(argv):
    if (len(argv) > 1):
        with open(argv[1]) as run_conf_file:
            conf = json.load(run_conf_file)
    else:
        conf = dict()

    tasks = conf["tasks"] if conf else ["TASK_1", "TASK_2"]
    n_prem = conf["num_premises"] if conf else 2
    max_distractors = conf["max_distractors"] if conf else 5
    subset_size = conf["subset_size"] if conf else 200
    batch_size = conf["batch_size"] if conf else 20
    icl = conf["icl"] if conf else False
    if (conf):
        models = conf["models"]
    else:
        models = [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-7b",
            "google/gemma-7b-it",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "BioMistral/BioMistral-7B",
        ]

    results = list()
    for task in tasks:
        for model in models:
            if (model == "mistralai/Mixtral-8x7B-Instruct-v0.1"):
                batch_size = 40
            else:
                batch_size = conf["batch_size"] if conf else 20
            for scheme in SyllogisticScheme:
                dummy_choices = [False, True] if (scheme.startswith("gen_modus")) else [False]
                for dummy in dummy_choices:
                    for variant in SyllogisticSchemeVariant:
                        exp = SyllogisticReasoningTest(model, scheme, variant, dummy=dummy, num_premises=n_prem,
                                                       batch_size=batch_size, subset_size=subset_size, icl=icl)
                        for n_distr in range(0, max_distractors + 1):
                            print("Running:", exp.conf_string(task, n_distr, False))
                            results.append(
                                {"scheme": scheme + (' (dummy)' if dummy else ''),
                                 "variant": variant,
                                 "model": model,
                                 "n_distractors": n_distr} | exp.run(task, num_distractors=n_distr)
                            )
                        del exp

        df_results = pd.DataFrame.from_records(results)
        print(df_results)
        df_results.to_csv(f"{task.lower()}{'-icl' if icl else ''}.tsv", sep="\t")



if __name__ == "__main__":
    main(sys.argv)
