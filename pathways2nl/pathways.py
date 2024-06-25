import random
import re
import pandas as pd
from typing import Tuple, List, Dict, Set
from enum import StrEnum, auto
from collections import Counter
from itertools import combinations, chain
from string import ascii_uppercase
from random import shuffle
from tqdm import tqdm
from treelib import Tree
from dynaconf import settings
from joblib import Parallel, delayed, cpu_count

MEMBER_RGX = re.compile(r"is a member of (?P<pw>.+) pathway")
ALL_MEMBER_RGX = re.compile(r"Every member of (?P<pw1>.+) pathway is( not)? a member of (?P<pw2>.+) pathway")
EXISTS_MEMBER_RGX = re.compile(r"There is at least one member of (?P<pw1>.+) pathway that is( not)? a member of (?P<pw2>.+) pathway")
DISJ_MEMBER_RGX = re.compile(r"Every member of (?P<pw1>.+) pathway is either a member of (?P<pwm1>.+) pathway or a member of (?P<pwm2>.+) pathway, or both")
GENE_RGX = re.compile(r"It is true that Gene (?P<gene>.+) is a member of")

STTM_TEMPLATES = {
    "all_member_incl": "Every member of {pw1} pathway is a member of {pw2} pathway",
    "all_member_notincl": "Every member of {pw1} pathway is not a member of {pw2} pathway",
    "all_member_disj": "Every member of {pw1} pathway is either a member of {pwm1} pathway or a member of {pwm2} pathway, or both",
    "exist_member_notincl": "There is at least one member of {pw1} pathway that is not a member of {pw2} pathway",
    "gene_incl": "Gene {gene} is a member of {pw} pathway",
    "gene_notincl": "Gene {gene} is not a member of {pw} pathway",
    "gene_hyp_incl": "It is true that Gene {gene} is a member of {pw} pathway",
    "gene_hyp_notincl": "It is true that Gene {gene} is not a member of {pw} pathway"
}


class SyllogisticScheme(StrEnum):
    GEN_MODUS_PONENS = auto()
    GEN_MODUS_TOLLENS = auto()
    GEN_CONTRAPOSITION = auto()
    HYPOTHETICAL_SYLLOGISM_1 = auto()
    HYPOTHETICAL_SYLLOGISM_3 = auto()
    DISJUNCTVE_SYLLOGISM = auto()
    GEN_DILEMMA = auto()


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
            pos_statements.append(STTM_TEMPLATES["gene_incl"].format(gene=gene, pw=pw_hier[0]))

        for j in range(len(pw_hier) - 1):
            pos_statements.append(STTM_TEMPLATES["all_member_incl"].format(pw1=pw_hier[j], pw2=pw_hier[j + 1]))
            for k in range(j + 2, len(pw_hier)):
                pos_statements.append(STTM_TEMPLATES["all_member_incl"].format(pw1=pw_hier[j], pw2=pw_hier[k]))

            for gene in genes:
                hipotheses.append(STTM_TEMPLATES["gene_hyp_incl"].format(gene=gene, pw=pw_hier[j + 1]))

            for pw_hier_oth, genes_oth in pw_hierarchy_genes:
                if (len(set(genes).intersection(genes_oth)) == 0):
                    neg_statements.append(f"No member of {pw_hier[0]} pathway is a member of {pw_hier_oth[0]} pathway")
                    for gene in genes:
                        neg_statements.append(STTM_TEMPLATES["gene_notincl"].format(gene=gene, pw=pw_hier_oth[0]))

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


def get_pw_genes(pathway: str, tree: Tree) -> Set[str]:
    genes = set()
    for node in tree.leaves(pathway):
        if (node.data):
            genes.update(node.data)

    return genes

def get_pw_intersect_sets(tree: Tree) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    pw_gene_sets = {pw.identifier: get_pw_genes(pw.identifier, tree) for pw in tree.all_nodes()}
    intersect_pw_sets = dict()
    for pw1 in pw_gene_sets:
        for pw2 in pw_gene_sets:
            if (pw1 != pw2 and (pw2, pw1) not in intersect_pw_sets):
                intersect_pw_sets[(pw1, pw2)] = pw_gene_sets[pw1].intersection(pw_gene_sets[pw2])

    return pw_gene_sets, intersect_pw_sets


def get_pw_chains(tree: Tree, n: int) -> List[Tuple[str]]:
    paths = [
        (tree.parent(node.identifier).identifier,) for node in tree.all_nodes()
        if (tree.parent(node.identifier) and tree.parent(node.identifier) != tree.root)
    ]
    for i in range(n - 1):
        paths = [
            p + (tree.parent(p[-1]).identifier,) for p in paths
            if (tree.parent(p[-1]) and tree.parent(p[-1]) != tree.root)
        ]

    return sorted(set(paths))

def get_premises(pw_chain: Tuple[str]) -> Dict[str, str]:
    return {
        f"P{i + 1}": STTM_TEMPLATES["all_member_incl"].format(pw1=pw_chain[i], pw2=pw_chain[i + 1])
        for i in range(len(pw_chain) - 1)
    }


def gen_modus_pt(tree: Tree, pw_chain: Tuple[str], dummy: bool, dummy_combs, ph_tuples: List[Dict[str, str]],
                 all_genes: Set[str], modus: str):
    genes = get_pw_genes(pw_chain[0], tree)

    if (modus == "ponens"):
        genes_set = genes
    else:  # Modus tollens
        genes_set = list(all_genes - genes)[:len(genes)]

    for gene in genes_set:
        gene_name = gene if not dummy else "".join(next(dummy_combs))
        premises = get_premises(pw_chain)

        if (modus == "ponens"):
            premises |= {f"P{len(premises) + 1}": STTM_TEMPLATES["gene_incl"].format(gene=gene_name, pw=pw_chain[0])}
            hypothesis = {"C": STTM_TEMPLATES["gene_hyp_incl"].format(gene=gene_name, pw=pw_chain[-1])}
        else:  # Modus tollens
            premises |= {f"P{len(premises) + 1}": STTM_TEMPLATES["gene_notincl"].format(gene=gene_name, pw=pw_chain[-1])}
            hypothesis = {"C": STTM_TEMPLATES["gene_hyp_notincl"].format(gene=gene_name, pw=pw_chain[0])}

        ph_tuples.append(premises | hypothesis)


def gen_contraposition(tree: Tree, pw_chain: Tuple[str], ph_tuples: List[Dict[str, str]], disjunct_size: int = 10):
    genes = get_pw_genes(pw_chain[0], tree)
    disjunct = [pw.identifier for pw in tree.all_nodes()
                if (len(get_pw_genes(pw.identifier, tree).intersection(genes)) == 0)]
    premises = get_premises(pw_chain)

    for disj_pw in disjunct[:disjunct_size]:
        disj_sttm = STTM_TEMPLATES["all_member_notincl"].format(pw1=pw_chain[-2], pw2=disj_pw)
        premises[sorted(premises.keys())[-1]] = disj_sttm
        hypothesis = {"C": STTM_TEMPLATES["all_member_notincl"].format(pw1=disj_pw, pw2=pw_chain[0])}

        ph_tuples.append(premises | hypothesis)


def hypothetical_syllogism(pw_chain: Tuple[str], ph_tuples: List[Dict[str, str]], mode: int):
    premises = get_premises(pw_chain)
    if (mode == 1):
        hypothesis = {"C": STTM_TEMPLATES["all_member_incl"].format(pw1=pw_chain[0], pw2=pw_chain[-1])}
    else:
        ex_sttm = STTM_TEMPLATES["exist_member_notincl"].format(pw1=pw_chain[-1], pw2=pw_chain[-2])
        premises[sorted(premises.keys())[-1]] = ex_sttm
        hypothesis = {"C": STTM_TEMPLATES["exist_member_notincl"].format(pw1=pw_chain[-1], pw2=pw_chain[0])}

    ph_tuples.append(premises | hypothesis)


def disjunctive_syllogism(tree: Tree) -> List[Dict[str, str]]:
    ph_tuples = list()
    pw_gene_sets, intersect_pw_sets = get_pw_intersect_sets(tree)

    for pw1 in pw_gene_sets:
        full_membership = [
            pw2 for pw2 in pw_gene_sets
            if (pw1 != pw2 and
                (pw1, pw2) in intersect_pw_sets and
                (len(intersect_pw_sets[(pw1, pw2)]) == len(pw_gene_sets[pw1])))
        ]
        no_membership = [
            pw2 for pw2 in pw_gene_sets
            if (pw1 != pw2 and
                (pw1, pw2) in intersect_pw_sets and
                len(intersect_pw_sets[(pw1, pw2)]) == 0)
        ]
        for pwm1, pwm2 in zip(no_membership, full_membership):
            ph_tuple = {
                "P1": STTM_TEMPLATES["all_member_disj"].format(pw1=pw1, pwm1=pwm1, pwm2=pwm2),
                "P2": STTM_TEMPLATES["all_member_notincl"].format(pw1=pw1, pw2=pwm1),
                "C": STTM_TEMPLATES["all_member_incl"].format(pw1=pw1, pw2=pwm2)
            }
            ph_tuples.append(ph_tuple)

    return ph_tuples


def generalised_dilemma(tree: Tree) -> List[Dict[str, str]]:
    ph_tuples = list()
    pw_gene_sets, intersect_pw_sets = get_pw_intersect_sets(tree)

    for pw1 in pw_gene_sets:
        full_membership = [
            pw2 for pw2 in pw_gene_sets
            if (pw1 != pw2 and
                (pw1, pw2) in intersect_pw_sets and
                (len(intersect_pw_sets[(pw1, pw2)]) == len(pw_gene_sets[pw1])))
        ]

        for pwm1 in full_membership:
            for pwm2 in full_membership:
                ext_membership = [
                    pwm3 for pwm3 in pw_gene_sets
                    if (pwm3 not in [pw1, pwm1, pwm2] and
                        (pwm1, pwm3) in intersect_pw_sets and
                        (pwm2, pwm3) in intersect_pw_sets and
                        (len(intersect_pw_sets[(pwm1, pwm3)]) == len(pw_gene_sets[pwm1])) and
                        (len(intersect_pw_sets[(pwm2, pwm3)]) == len(pw_gene_sets[pwm2])))
                ]

                for pwm3 in ext_membership:
                    ph_tuple = {
                        "P1": STTM_TEMPLATES["all_member_disj"].format(pw1=pw1, pwm1=pwm1, pwm2=pwm2),
                        "P2": STTM_TEMPLATES["all_member_incl"].format(pw1=pwm1, pw2=pwm3),
                        "P3": STTM_TEMPLATES["all_member_incl"].format(pw1=pwm2, pw2=pwm3),
                        "C": STTM_TEMPLATES["all_member_incl"].format(pw1=pw1, pw2=pwm3)
                    }
                    ph_tuples.append(ph_tuple)

    return ph_tuples


def get_ph_tuples(tree: Tree, scheme: SyllogisticScheme, length: int = 2, dummy: bool = False) -> List[Dict[str, str]]:
    ph_tuples = list()

    if (scheme == SyllogisticScheme.DISJUNCTVE_SYLLOGISM):
        ph_tuples = disjunctive_syllogism(tree)
    elif (scheme == SyllogisticScheme.GEN_DILEMMA):
        ph_tuples = generalised_dilemma(tree)
    else:
        effective_length = length if ("modus" in scheme) else length + 1
        pw_chains = get_pw_chains(tree, effective_length)
        dummy_combs = combinations(ascii_uppercase, r=4)


        all_genes = set()
        for node in tree.leaves(tree.root):
            if (node.data):
                all_genes.update(node.data)

        for pw_chain in pw_chains:
            if (scheme == SyllogisticScheme.GEN_MODUS_PONENS):
                gen_modus_pt(tree, pw_chain, dummy, dummy_combs, ph_tuples, all_genes, "ponens")
            elif (scheme == SyllogisticScheme.GEN_MODUS_TOLLENS):
                gen_modus_pt(tree, pw_chain, dummy, dummy_combs, ph_tuples, all_genes, "tollens")
            elif (scheme == SyllogisticScheme.GEN_CONTRAPOSITION):
                gen_contraposition(tree, pw_chain, ph_tuples)
            elif (scheme.startswith("hypothetical_syllogism")):
                mode = int(scheme.split("_")[-1])
                hypothetical_syllogism(pw_chain, ph_tuples, mode)

    return ph_tuples


def falsify(ph_tuples: List[Dict[str, str]]):
    falsified_tuples = list()

    for ph_tuple in ph_tuples:
        falsified_tuples.append(ph_tuple.copy())
        if (falsified_tuples[-1]["C"].startswith("Every")):
            falsified_tuples[-1]["C"] = falsified_tuples[-1]["C"].replace("Every", "Not every")
        else:
            falsified_tuples[-1]["C"] = falsified_tuples[-1]["C"].replace("It is true", "It is false")

    return falsified_tuples


def add_distractors(ph_tuple: Dict[str, str], oth_ph_tuples: List[Dict[str, str]], tree: Tree, n: int):
    premise_keys = [pk for pk in ph_tuple if pk != "C"]
    genes = set()
    for p_key in premise_keys:
        if ("Gene" not in ph_tuple[p_key]):
            mo = ALL_MEMBER_RGX.search(ph_tuple[p_key])
            if (not mo):
                mo = EXISTS_MEMBER_RGX.search(ph_tuple[p_key])
            if (not mo):
                mo = DISJ_MEMBER_RGX.search(ph_tuple[p_key])
            if (not mo):
                print("MISSED RGX:", ph_tuple[p_key])
                exit(1)
            for pw in [mo.group(pwk) for pwk in mo.groupdict() if pwk.startswith("pw")]:
                if (pw != tree.root):
                    genes.update(chain(*[nd.data for nd in tree.leaves(pw)]))

    unrel_filter = lambda node: len(
        genes.intersection(set(chain(*[nd.data for nd in tree.leaves(node.identifier)])))) == 0
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

    return ph_tuple


def add_distractors_all(ph_tuples: List[Dict[str, str]], tree: Tree, n: int = 5, seed: int = None):
    oth_ph_tuples = list(ph_tuples)
    random.seed(seed)
    shuffle(oth_ph_tuples)

    with Parallel(n_jobs=cpu_count(True)) as ppool:
        upd_ph_tuples = ppool(delayed(add_distractors)(ph_tuple, oth_ph_tuples, tree, n)
                              for ph_tuple in tqdm(ph_tuples, desc="Adding distractors"))

    for i in range(len(ph_tuples)):
        for key in upd_ph_tuples[i]:
            ph_tuples[i][key] = upd_ph_tuples[i][key]




