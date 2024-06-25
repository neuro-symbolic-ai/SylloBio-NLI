import json
import pandas as pd
from pathways2nl.pathways import load_hierarchy_genes, get_tree, get_ph_tuples, falsify, add_distractors, SyllogisticScheme
from pathways2nl.experiments import SyllogisticReasoningTest


def gen_datasets(scheme: SyllogisticScheme):
    pw_hierarchy_genes = load_hierarchy_genes()

    # pos_statements, neg_statements, hipotheses = gen_statements(pw_hierarchy_genes)
    # print(len(pos_statements), len(neg_statements), len(hipotheses))
    # with open("pos_statements.txt", "w") as out_file:
    #     print("\n".join(pos_statements), file=out_file)
    #
    # with open("neg_statements.txt", "w") as out_file:
    #     print("\n".join(neg_statements), file=out_file)
    #
    # with open("hipotheses.txt", "w") as out_file:
    #     print("\n".join(hipotheses), file=out_file)

    # with open("statements_by_gene.json", "w") as out_file:
    #     statements = pos_statements + neg_statements + hipotheses
    #     json.dump({gene: get_statements_by_gene(gene, statements) for gene in ["PI3K"]},
    #               out_file, indent=2)

    tree = get_tree(pw_hierarchy_genes)
    ph_tuples = get_ph_tuples(tree, scheme, 2)
    dummy_ph_tuples = get_ph_tuples(tree, scheme, 2, dummy=True)

    with open("outputs/sets/gen_modus_ponens_l2.json", "w") as set1_file:
        json.dump(ph_tuples, set1_file, indent=2)

    with open("outputs/sets/falsified_gen_modus_ponens_l2.json", "w") as set2_file:
        json.dump(falsify(ph_tuples), set2_file, indent=2)

    add_distractors(ph_tuples, tree, seed=0)
    with open("outputs/sets/distrac_modus_ponens_l2.json", "w") as set3_file:
        json.dump(ph_tuples, set3_file, indent=2)

    with open("outputs/sets/distrac_falsified_modus_ponens_l2.json", "w") as set4_file:
        json.dump(falsify(ph_tuples), set4_file, indent=2)

    with open("outputs/sets/gen_modus_ponens_l2_dummy.json", "w") as set1_dummy_file:
        json.dump(dummy_ph_tuples, set1_dummy_file, indent=2)

    with open("outputs/sets/falsified_gen_modus_ponens_l2_dummy.json", "w") as set2_dummy_file:
        json.dump(falsify(dummy_ph_tuples), set2_dummy_file, indent=2)

    add_distractors(dummy_ph_tuples, tree, seed=0)
    with open("outputs/sets/distrac_modus_ponens_l2_dummy.json", "w") as set3_dummy_file:
        json.dump(dummy_ph_tuples, set3_dummy_file, indent=2)

    with open("outputs/sets/distrac_falsified_modus_ponens_l2_dummy.json", "w") as set4_dummy_file:
        json.dump(falsify(dummy_ph_tuples), set4_dummy_file, indent=2)

    print(tree.show(stdout=False))


def main():
    # gen_datasets()
    models = [
        # "mistralai/Mistral-7B-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "google/gemma-7b",
        # "google/gemma-7b-it",
        # "meta-llama/Meta-Llama-3-8B",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "BioMistral/BioMistral-7B",
        # "NousResearch/Hermes-2-Pro-Llama-3-8B",
    ]

    results = list()
    for task in ["TASK_1"]:
        for dummy in [False, True]:
            for model in models:
                for scheme in SyllogisticScheme:
                    exp = SyllogisticReasoningTest(model, scheme, dummy=dummy, num_premises=2, batch_size=10)
                    for n_distr in range(0, 6):
                        results.append(
                            {"scheme": scheme, "model": model, "n_distractors": n_distr} | exp.run(task, subset_size=200, num_distractors=n_distr)
                        )
                    del exp

            df_results = pd.DataFrame.from_records(results)
            print(df_results)
            df_results.to_csv(f"{task.lower()}{'-dummy' if dummy else ''}.tsv", sep="\t")





if __name__ == "__main__":
    main()
