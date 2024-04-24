import pandas as pd
from typing import Tuple, List
from collections import Counter
from tqdm import tqdm


def load_hierarchy_genes():
    df = pd.read_excel("data/reactome_pathways_API.xlsx", dtype={"Length hierarchy": int})
    df = df[df["Length hierarchy"] <= 5]
    pw_hierarchy_names = [eval(pwhn) for pwhn in df["pathways_hierarchy_names"].tolist()]
    pw_genes = [(eval(genes) if (genes.strip()) else []) for genes in df["Genes"].fillna("").tolist()]
    pw_hierarchy_genes = [(hier, genes) for hier, genes in zip(pw_hierarchy_names, pw_genes) if (genes)]

    return pw_hierarchy_genes


def gen_statements(pw_hierarchy_genes: List[Tuple[List[str], List[str]]]):
    pos_statements = list()
    neg_statements = list()
    hipotheses = list()
    for pw_hier, genes in tqdm(pw_hierarchy_genes, desc="Generating statements"):
        for gene in genes:
            pos_statements.append(f"Gene {gene} is a member of {pw_hier[0]} pathway")
        for j in range(len(pw_hier) - 1):
            statement = f"Every member of {pw_hier[j]} pathway is a member of {pw_hier[j + 1]} pathway"
            pos_statements.append(statement)
            for k in range(j + 2, len(pw_hier)):
                statement = f"Every member of {pw_hier[j]} pathway is a member of {pw_hier[k]} pathway"
                pos_statements.append(statement)
            for gene in genes:
                hipotheses.append(f"It is true that Gene {gene} is a member of {pw_hier[j + 1]} pathway")

            for pw_hier_oth, genes_oth in pw_hierarchy_genes:
                if (len(set(genes).intersection(genes_oth)) == 0):
                    neg_statements.append(f"No member of {pw_hier[0]} pathway is a member of {pw_hier_oth[0]} pathway")
                    for gene in genes:
                        neg_statements.append(f"Gene {gene} is not a member of {pw_hier_oth[0]} pathway")

    return list(Counter(pos_statements).keys()), list(Counter(neg_statements).keys()), list(Counter(hipotheses).keys())

def main():
    pw_hierarchy_genes = load_hierarchy_genes()
    pos_statements, neg_statements, hipotheses = gen_statements(pw_hierarchy_genes)
    print(len(pos_statements), len(neg_statements), len(hipotheses))
    with open("pos_statements.txt", "w") as out_file:
        print("\n".join(pos_statements), file=out_file)

    with open("neg_statements.txt", "w") as out_file:
        print("\n".join(neg_statements), file=out_file)

    with open("hipotheses.txt", "w") as out_file:
        print("\n".join(hipotheses), file=out_file)


if __name__ == "__main__":
    main()