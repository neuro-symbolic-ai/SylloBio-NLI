import json
import pandas as pd
from pathways2nl.pathways import SyllogisticScheme, SyllogisticSchemeVariant
from pathways2nl.experiments import SyllogisticReasoningTest


def main():
    # gen_datasets()
    models = [
        # "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "google/gemma-7b",
        # "google/gemma-7b-it",
        # "meta-llama/Meta-Llama-3-8B",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "BioMistral/BioMistral-7B",
    ]

    results = list()
    for task in ["TASK_1"]:
        for model in models:
            for scheme in SyllogisticScheme:
                dummy_choices = [False, True] if (scheme.startswith("gen_modus")) else [False]
                for dummy in dummy_choices:
                    for variant in SyllogisticSchemeVariant:
                        exp = SyllogisticReasoningTest(model, scheme, variant, dummy=dummy, num_premises=2, batch_size=20)
                        for n_distr in range(0, 6):
                            print("Running:", exp.conf_string(task, n_distr, False))
                            results.append(
                                {"scheme": scheme, "model": model, "n_distractors": n_distr} | exp.run(task, num_distractors=n_distr)
                            )
                        del exp

            df_results = pd.DataFrame.from_records(results)
            print(df_results)
            df_results.to_csv(f"{task.lower()}{'-dummy' if dummy else ''}.tsv", sep="\t")





if __name__ == "__main__":
    main()
