import random
import re
import pandas as pd
from typing import Tuple, List, Dict
from collections import Counter
from itertools import combinations, chain
from string import ascii_uppercase
from random import shuffle
from tqdm import tqdm
from treelib import Tree
from dynaconf import settings

MEMBER_RGX = re.compile(r"is a member of (?P<pw>.+) pathway")
ALL_MEMBER_RGX = re.compile(r"Every member of (?P<pw1>.+) pathway is a member of (?P<pw2>.+) pathway")
GENE_RGX = re.compile(r"It is true that Gene (?P<gene>.+) is a member of")

def load_hierarchy_genes(data_path: str = settings["data_file_path"]):
    df = pd.read_excel(data_path, dtype={"Length hierarchy": int})
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
            pos_statements.append(f"Every member of {pw_hier[j]} pathway is a member of {pw_hier[j + 1]} pathway")
            for k in range(j + 2, len(pw_hier)):
                pos_statements.append(f"Every member of {pw_hier[j]} pathway is a member of {pw_hier[k]} pathway")

            for gene in genes:
                hipotheses.append(f"It is true that Gene {gene} is a member of {pw_hier[j + 1]} pathway")

            for pw_hier_oth, genes_oth in pw_hierarchy_genes:
                if (len(set(genes).intersection(genes_oth)) == 0):
                    neg_statements.append(f"No member of {pw_hier[0]} pathway is a member of {pw_hier_oth[0]} pathway")
                    for gene in genes:
                        neg_statements.append(f"Gene {gene} is not a member of {pw_hier_oth[0]} pathway")

    pos_statements = list(Counter(pos_statements).keys())
    neg_statements = list(Counter(neg_statements).keys())
    hipotheses = list(Counter(hipotheses).keys())

    return pos_statements, neg_statements, hipotheses


def get_statements_by_gene(gene: str, statements: List[str]):
    gene_sttms = [
        sttm for sttm in statements
        if ((sttm.startswith("Gene") or sttm.startswith("It is")) and gene in sttm)
    ]
    pathways = set([MEMBER_RGX.search(sttm) for sttm in gene_sttms])
    member_sttms = Counter()
    for pw in pathways:
        member_sttms.update([
            sttm for sttm in statements
            if (pw and not(sttm.startswith("Gene") or sttm.startswith("It is")) and pw.group("pw") in sttm)
        ])

    return gene_sttms + list(member_sttms.keys())


def get_tree(pw_hierarchy_genes: List[Tuple[List[str], List[str]]]):
    tree = Tree()
    for pw_hier, genes in tqdm(pw_hierarchy_genes, desc="Loading pathway tree"):
        parent = None
        if (pw_hier[-1].strip() == "Disease"):
            for pw in reversed(pw_hier):
                pw = pw.strip()
                if (not tree.contains(pw)):
                    tree.create_node(pw, pw, parent)
                parent = pw
            leaf = tree.get_node(pw_hier[0].strip())
            if (not leaf.data):
                leaf.data = list()
            leaf.data.extend(genes)

    return tree


def get_ph_tuples(tree: Tree, length: int = 2, include_genes: bool = True, dummy: bool = False) -> List[Dict[str, str]]:
    sttm_stack = list()
    ph_tuples = list()
    pw_stack = [tree.root]
    visited = set()
    dummy_combs = combinations(ascii_uppercase, r=4)
    effective_length = ((length - 1) if include_genes else length)
    while (pw_stack):
        cur_node = pw_stack.pop()
        if (cur_node in visited):
            continue
        visited.add(cur_node)
        if (tree.parent(cur_node)):
            sttm_stack.append(f"Every member of {cur_node} pathway is a member of {tree.parent(cur_node).identifier} pathway")
            if (len(sttm_stack) >= effective_length):
                if (include_genes):
                    genes = set()
                    for node in tree.leaves(cur_node):
                        if (node.data):
                            genes.update(node.data)
                    for gene in genes:
                        gene_name = gene if not dummy else "".join(next(dummy_combs))
                        premises = {f"P{i + 1}": sttm for i, sttm in enumerate(reversed(sttm_stack[-effective_length:]))}
                        premises |= {f"P{length}": f"Gene {gene_name} is a member of {cur_node} pathway"}
                        pw_hyp = ALL_MEMBER_RGX.search(sttm_stack[-effective_length:][0]).group("pw2")
                        hypothesis = {"C": f"It is true that Gene {gene_name} is a member of {pw_hyp} pathway"}
                        ph_tuples.append(premises | hypothesis)
                else:
                    premises = {f"P{i + 1}": sttm for i, sttm in enumerate(reversed(sttm_stack[-effective_length:]))}
                    pw_hyp1 = ALL_MEMBER_RGX.search(sttm_stack[-effective_length:][-1]).group("pw1")
                    pw_hyp2 = ALL_MEMBER_RGX.search(sttm_stack[-effective_length:][0]).group("pw2")
                    hypothesis = {"C": f"Every member of {pw_hyp1} pathway is a member of {pw_hyp2} pathway"}
                    ph_tuples.append(premises | hypothesis)

        for node in reversed(tree.children(cur_node)):
            pw_stack.append(node.identifier)

        if (len(tree.children(cur_node)) == 0):
            sttm_stack.pop()
            if (tree.parent(cur_node)):
                parent = tree.parent(cur_node).identifier
                while (parent):
                    siblings = [node.identifier for node in tree.children(parent)]
                    if (len(visited.intersection(siblings)) == len(siblings) and len(sttm_stack) > 0):
                        sttm_stack.pop()
                    else:
                        break
                    parent = tree.parent(parent).identifier

    return ph_tuples


def falsify(ph_tuples: List[Dict[str, str]]):
    falsified_tuples = list()

    for ph_tuple in ph_tuples:
        falsified_tuples.append(ph_tuple.copy())
        falsified_tuples[-1]["C"] = falsified_tuples[-1]["C"].replace("It is true", "It is false")

    return falsified_tuples


def add_distractors(ph_tuples: List[Dict[str, str]], tree: Tree, n: int = 5, seed: int = None):
    oth_ph_tuples = list(ph_tuples)
    random.seed(seed)
    shuffle(oth_ph_tuples)

    for ph_tuple in tqdm(ph_tuples, desc="Adding distractors"):
        premise_keys = [pk for pk in ph_tuple if pk != "C"]
        genes = set()
        for p_key in premise_keys:
            if ("Gene" not in ph_tuple[p_key]):
                mo = ALL_MEMBER_RGX.search(ph_tuple[p_key])
                for pw in [mo.group("pw1"), mo.group("pw2")]:
                    if (pw != tree.root):
                        genes.update(chain(*[nd.data for nd in tree.leaves(pw)]))

        unrel_filter = lambda node: len(genes.intersection(set(chain(*[nd.data for nd in tree.leaves(node.identifier)])))) == 0
        unrel_pws = [node.identifier for node in tree.filter_nodes(unrel_filter)]

        i = 0
        while (i < n):
            for oth_ph_tuple in oth_ph_tuples:
                if (oth_ph_tuple["C"] == ph_tuple["C"]):
                    continue
                for sk in oth_ph_tuple:
                    if (sk != "C" and "Gene" not in oth_ph_tuple[sk] and oth_ph_tuple[sk] not in set(ph_tuple.items())):
                        unrel = sum([f" {pw} " in oth_ph_tuple[sk] for pw in unrel_pws]) > 0
                        if (unrel):
                            ph_tuple[f"P{len(premise_keys) + 1 + i}"] = oth_ph_tuple[sk]
                            i += 1
                            if (i >= n):
                                break
                if (i >= n):
                    break

        # Puts hypothesis / conclusion back at the end of the dict structure.
        hyp = ph_tuple["C"]
        del ph_tuple["C"]
        ph_tuple["C"] = hyp


