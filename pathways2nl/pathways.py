import json
import random
import re
import pandas as pd
from typing import Tuple, List, Dict, Set, Optional, Iterable, Self
from enum import StrEnum, auto
from itertools import combinations, permutations
from string import ascii_uppercase
from random import shuffle
from tqdm import tqdm
from treelib import Tree
from dynaconf import settings
from joblib import Parallel, delayed, cpu_count
from .logic2nl import Logic2NLTranslator

MEMBER_RGX = re.compile(r"member of (?P<pw>.+?) pathway")


class SyllogisticScheme(StrEnum):
    GEN_MODUS_PONENS = auto()
    GEN_MODUS_TOLLENS = auto()
    GEN_CONTRAPOSITION = auto()
    HYPOTHETICAL_SYLLOGISM_1 = auto()
    HYPOTHETICAL_SYLLOGISM_3 = auto()
    DISJUNCTIVE_SYLLOGISM = auto()
    GEN_DILEMMA = auto()


class SyllogisticSchemeVariant(StrEnum):
    BASE = auto()
    NEGATION = auto()
    COMPLEX_PREDICATES = auto()
    DE_MORGAN = auto()


class Pathway(Set[str]):
    def __init__(self, name: str, initial: Iterable[str]):
        super(Pathway, self).__init__(initial)
        self.name: str = name

    def __repr__(self):
        return json.dumps({"name": self.name, "genes": list(self)[:10] + (["..."] if len(self) > 10 else [])})

    @staticmethod
    def conjunction(*args):
        """Pathway conjunction: intersection of the respective gene sets"""
        conj_name = " ∧ ".join([(arg.name if isinstance(arg, Pathway) else "?") for arg in args])
        return Pathway(conj_name, set.intersection(*args))

    @staticmethod
    def disjunction(*args):
        """Pathway disjuction: union of the respective gene sets"""
        conj_name = " v ".join([(arg.name if isinstance(arg, Pathway) else "?") for arg in args])
        return Pathway(conj_name, set.union(*args))

    def negation(self, *args) -> Self:
        """Pathway negation: complement of the respective gene set"""
        universe = set.union(*args)
        return Pathway(f"¬({self.name})", universe - self)

    def implies(self, other: Self):
        """Tests pw1(x) -> pw2(x)"""
        return self.issubset(other)

    def excludes(self, other: Self):
        """Tests pw1(x) -> ¬pw2(x)"""
        return self.isdisjoint(other)


class PathwaySetOps:
    HIER_SCHEMES = set(SyllogisticScheme) - {SyllogisticScheme.DISJUNCTIVE_SYLLOGISM, SyllogisticScheme.GEN_DILEMMA}

    def __init__(self, tree: Optional[Tree] = None):
        self.tree = tree
        if (not tree):
            self.tree = PathwaySetOps.get_tree(PathwaySetOps.load_hierarchy_genes())
        self.all_genes = set()
        for node in self.tree.leaves(self.tree.root):
            if (node.data):
                self.all_genes.update(node.data)
        self.transl = Logic2NLTranslator()

        with open(settings["scheme_template_file_path"]) as schemes_file:
            self.scheme_templates = json.load(schemes_file)

    @staticmethod
    def load_hierarchy_genes(data_path: str = settings["data_file_path"]):
        df = pd.read_excel(data_path, dtype={"Length hierarchy": int})
        df = df[df["Length hierarchy"] <= 5]
        pw_hierarchy_names = [eval(pwhn) for pwhn in df["pathways_hierarchy_names"].tolist()]
        pw_genes = [(eval(genes) if (genes.strip()) else []) for genes in df["Genes"].fillna("").tolist()]
        pw_hierarchy_genes = [(hier, genes) for hier, genes in zip(pw_hierarchy_names, pw_genes) if (genes)]

        return pw_hierarchy_genes

    @staticmethod
    def get_tree(pw_hierarchy_genes: List[Tuple[List[str], List[str]]]) -> Tree:
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

    def get_pathway(self, pathway_name: str) -> Pathway:
        pathway = Pathway(pathway_name, [])
        for node in self.tree.leaves(pathway_name):
            if (node.data):
                pathway.update(node.data)

        return pathway

    def get_all_pathways(self) -> List[Pathway]:
        return [self.get_pathway(node.identifier) for node in self.tree.all_nodes()]

    def get_pw_chains(self, chain_length: int, scheme: SyllogisticScheme,
                      variant: SyllogisticSchemeVariant, subset_size: int) -> List[Tuple[Pathway, ...]]:
        pw_chains = list()
        pathways = list(reversed(self.get_all_pathways()))
        if (scheme in PathwaySetOps.HIER_SCHEMES):
            paths = [
                (self.tree.parent(node.identifier).identifier,) for node in self.tree.all_nodes()
                if (self.tree.parent(node.identifier) and self.tree.parent(node.identifier) != self.tree.root)
            ]
            for i in range(chain_length - 1):
                paths = [
                    p + (self.tree.parent(p[-1]).identifier,) for p in paths
                    if (self.tree.parent(p[-1]) and self.tree.parent(p[-1]) != self.tree.root)
                ]

            pw_chains_hier = [tuple([self.get_pathway(pw_name) for pw_name in pw_chain]) for pw_chain in sorted(set(paths))]

            if (scheme in [SyllogisticScheme.GEN_MODUS_PONENS, SyllogisticScheme.GEN_MODUS_TOLLENS]):
                if (variant == SyllogisticSchemeVariant.BASE):
                    pw_chains = pw_chains_hier
                if (variant == SyllogisticSchemeVariant.NEGATION):
                    for pwc in pw_chains_hier:
                        pw_chains.extend([pwc[:-1] + (pw,) for pw in pathways if pwc[-2].excludes(pw)][:2])
                elif (variant in ["complex_predicates", "de_morgan"]):
                    if (scheme == SyllogisticScheme.GEN_MODUS_PONENS):
                        if (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
                            for pw_f, pw_h in combinations(pathways, 2):
                                left_term = Pathway.conjunction(pw_f, pw_h)
                                pw_chains.extend([(pw_f, pw_g, pw_h) for pw_g in pathways
                                                  if (pw_g.name != pw_f.name and
                                                      pw_g.name != pw_h.name and
                                                      left_term.implies(pw_g))])
                                if (len(pw_chains) >= subset_size):
                                    break
                        else:  # De Morgan variant
                            for pw_f, pw_h in combinations(pathways, 2):
                                left_term = Pathway.disjunction(pw_f, pw_h).negation(self.all_genes)
                                pw_chains.extend([(pw_f, pw_g, pw_h) for pw_g in pathways
                                                  if (pw_g.name != pw_f.name and
                                                      pw_g.name != pw_h.name and
                                                      left_term.implies(pw_g))])
                                if (len(pw_chains) >= subset_size):
                                    break
                    else:  # Modus tollens
                        for pw_g, pw_h in combinations(pathways, 2):
                            right_term = Pathway.conjunction(pw_g, pw_h)
                            pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.implies(right_term))])
                            if (len(pw_chains) >= subset_size):
                                break
            elif (scheme == SyllogisticScheme.GEN_CONTRAPOSITION):
                if (variant == SyllogisticSchemeVariant.BASE):
                    for pwc in pw_chains_hier:
                        pw_chains.extend([pwc[:-1] + (pw,) for pw in pathways if pwc[-2].excludes(pw)][:2])
                elif (variant == SyllogisticSchemeVariant.NEGATION):
                    pw_chains = pw_chains_hier
                elif (variant in ["complex_predicates", "de_morgan"]):
                    for pw_f, pw_h in combinations(pathways, 2):
                        left_term = Pathway.conjunction(pw_f, pw_h)
                        pw_chains.extend([(pw_f, pw_g, pw_h) for pw_g in pathways
                                          if (pw_g.name != pw_f.name and
                                              pw_g.name != pw_h.name and
                                              left_term.excludes(pw_g))])
                        if (len(pw_chains) >= subset_size):
                            break
            elif (scheme == SyllogisticScheme.HYPOTHETICAL_SYLLOGISM_1):
                if (variant == SyllogisticSchemeVariant.BASE):
                    pw_chains = pw_chains_hier
                if (variant == SyllogisticSchemeVariant.NEGATION):
                    for pw_g, pw_h in combinations(pathways, 2):
                        sec_prop = pw_g.negation(self.all_genes).implies(pw_h)
                        if (sec_prop):
                            pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.excludes(pw_g))])
                            if (len(pw_chains) >= subset_size):
                                break
                elif (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
                    for pw_g, pw_i in combinations(pathways, 2):
                        conj_gi = Pathway.conjunction(pw_g, pw_i)
                        pw_chains.extend([(pw_f, pw_g, pw_h, pw_i) for pw_f, pw_h in permutations(pathways, 2)
                                          if (pw_f.name != pw_g.name and pw_f.name != pw_i.name and
                                              pw_h.name != pw_g.name and pw_h.name != pw_i.name and
                                              pw_f.implies(pw_g) and pw_f.implies(pw_i) and conj_gi.implies(pw_h))])
                        if (len(pw_chains) >= subset_size):
                            break
                elif (variant == SyllogisticSchemeVariant.DE_MORGAN):
                    for pw_f, pw_i in combinations(pathways, 2):
                        conj_neg_fi = Pathway.conjunction(pw_f.negation(self.all_genes), pw_i.negation(self.all_genes))
                        pw_chains.extend([(pw_f, pw_g, pw_h, pw_i) for pw_g, pw_h in permutations(pathways, 2)
                                          if (pw_g.name != pw_f.name and pw_g.name != pw_i.name and
                                              pw_h.name != pw_f.name and pw_h.name != pw_i.name and
                                              conj_neg_fi.implies(pw_g) and pw_g.implies(pw_h))])
                        if (len(pw_chains) >= subset_size):
                            break
            elif (scheme == SyllogisticScheme.HYPOTHETICAL_SYLLOGISM_3):
                if (variant == SyllogisticSchemeVariant.BASE):
                    for pw_h, pw_g in combinations(pathways, 2):
                        sec_prop = len(Pathway.conjunction(pw_h, pw_g.negation(self.all_genes))) > 0
                        if (sec_prop):
                            pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.implies(pw_g))])
                            if (len(pw_chains) >= subset_size):
                                break
                elif (variant == SyllogisticSchemeVariant.NEGATION):
                    for pw_h, pw_g in combinations(pathways, 2):
                        sec_prop = len(Pathway.conjunction(pw_h, pw_g.negation(self.all_genes))) > 0
                        if (sec_prop):
                            pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.negation(self.all_genes).implies(pw_g))])
                            if (len(pw_chains) >= subset_size):
                                break
                elif (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
                    for pw_g, pw_i in combinations(pathways, 2):
                        neg_conj_gi = Pathway.conjunction(pw_g, pw_i).negation(self.all_genes)
                        for pw_h in pathways:
                            if (pw_h.name != pw_g.name and pw_h.name != pw_i.name):
                                thr_prop = len(Pathway.conjunction(pw_h, neg_conj_gi)) > 0
                                if (thr_prop):
                                    pw_chains.extend([(pw_f, pw_g, pw_h, pw_i) for pw_f in pathways
                                                     if (pw_f.name != pw_g.name and
                                                         pw_f.name != pw_h.name and
                                                         pw_f.name != pw_i.name and
                                                         pw_f.implies(pw_g) and pw_f.implies(pw_i))])
                                    if (len(pw_chains) >= subset_size):
                                        break
                        if (len(pw_chains) >= subset_size):
                            break

                else:  # De Morgan variant
                    for pw_g, pw_i in combinations(pathways, 2):
                        conj_neg_gi = Pathway.disjunction(pw_g.negation(self.all_genes), pw_i.negation(self.all_genes))
                        for pw_h in pathways:
                            if (pw_h.name != pw_g.name and pw_h.name != pw_i.name):
                                thr_prop = len(Pathway.conjunction(pw_h, conj_neg_gi)) > 0
                                if (thr_prop):
                                    pw_chains.extend([(pw_f, pw_g, pw_h, pw_i) for pw_f in pathways
                                                     if (pw_f.name != pw_g.name and
                                                         pw_f.name != pw_h.name and
                                                         pw_f.name != pw_i.name and
                                                         pw_f.implies(pw_g) and pw_f.implies(pw_i))])
                                    if (len(pw_chains) >= subset_size):
                                        break
                        if (len(pw_chains) >= subset_size):
                            break

        elif (scheme == SyllogisticScheme.DISJUNCTIVE_SYLLOGISM):
            if (variant == SyllogisticSchemeVariant.BASE):
                for pw_g, pw_h in combinations(pathways, 2):
                    disj_gh = Pathway.disjunction(pw_g, pw_h)
                    pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                      if (pw_f.name != pw_g.name and
                                          pw_f.name != pw_h.name and
                                          pw_f.implies(disj_gh) and pw_f.excludes(pw_g))])
                    if (len(pw_chains) >= subset_size):
                        break

            elif (variant == SyllogisticSchemeVariant.NEGATION):
                for pw_g, pw_h in combinations(pathways, 2):
                    disj_gh = Pathway.disjunction(pw_g, pw_h)
                    pw_chains.extend([(pw_f, pw_g, pw_h) for pw_f in pathways
                                      if (pw_f.name != pw_g.name and
                                          pw_f.name != pw_h.name and
                                          pw_f.implies(disj_gh) and pw_g.excludes(pw_f))])
                    if (len(pw_chains) >= subset_size):
                        break

            elif (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
                for pw_g, pw_h, pw_i in combinations(pathways, 3):
                    conj_ghi = Pathway.disjunction(pw_g, pw_h, pw_i)
                    pw_chains.extend([(pw_f, pw_g, pw_h, pw_i) for pw_f in pathways
                                      if (pw_f.name != pw_g.name and
                                          pw_f.name != pw_h.name and
                                          pw_f.name != pw_i.name and
                                          pw_f.implies(conj_ghi) and pw_f.excludes(pw_g) and pw_f.excludes(pw_i))])
                    if (len(pw_chains) >= subset_size):
                        break

            else:  # De Morgan variant
                for pw_f, pw_i in combinations(pathways, 2):
                    conj_fi = Pathway.conjunction(pw_f, pw_i)
                    disj_neg_fi = Pathway.disjunction(pw_f.negation(self.all_genes), pw_i.negation(self.all_genes))
                    for pw_g, pw_h in combinations(pathways, 2):
                        if (pw_g.name != pw_f.name and pw_g.name != pw_i.name and
                            pw_h.name != pw_f.name and pw_h.name != pw_i.name):
                            disj_gh = Pathway.disjunction(pw_g, pw_h)
                            if (conj_fi.implies(disj_gh) and pw_g.implies(disj_neg_fi)):
                                pw_chains.append((pw_f, pw_g, pw_h, pw_i))
                                if (len(pw_chains) >= subset_size):
                                    break
                    if (len(pw_chains) >= subset_size):
                        break

        elif (scheme == SyllogisticScheme.GEN_DILEMMA):
            if (variant == SyllogisticSchemeVariant.BASE):
                for pw_g, pw_h in combinations(pathways, 2):
                    disj_gh = Pathway.disjunction(pw_g, pw_h)
                    for pw_j in pathways:
                        sec_prop = pw_g.implies(pw_j)
                        thr_prop = pw_h.implies(pw_j)
                        if (pw_j.name != pw_g.name and pw_j.name != pw_h.name and sec_prop and thr_prop):
                            pw_chains.extend([(pw_f, pw_g, pw_h, pw_j) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.name != pw_j.name and
                                                  pw_f.implies(disj_gh))])
                            if (len(pw_chains) >= subset_size):
                                break
                    if (len(pw_chains) >= subset_size):
                        break

            elif (variant == SyllogisticSchemeVariant.NEGATION):
                for pw_g, pw_h in combinations(pathways, 2):
                    disj_gh = Pathway.disjunction(pw_g, pw_h)
                    for pw_j in pathways:
                        sec_prop = pw_j.excludes(pw_g)
                        thr_prop = pw_j.excludes(pw_h)
                        if (pw_j.name != pw_g.name and pw_j.name != pw_h.name and sec_prop and thr_prop):
                            pw_chains.extend([(pw_f, pw_g, pw_h, pw_j) for pw_f in pathways
                                              if (pw_f.name != pw_g.name and
                                                  pw_f.name != pw_h.name and
                                                  pw_f.name != pw_j.name and
                                                  pw_f.implies(disj_gh))])
                            if (len(pw_chains) >= subset_size):
                                break
                    if (len(pw_chains) >= subset_size):
                        break

            elif (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
                for pw_g, pw_h, pw_i in combinations(pathways, 3):
                    disj_ghi = Pathway.disjunction(pw_g, pw_h, pw_i)
                    for pw_f in pathways:
                        if (pw_f.name != pw_g.name and pw_f.name != pw_h.name and
                            pw_f.name != pw_i.name and pw_f.implies(disj_ghi)):
                            for pw_j in pathways:
                                if (pw_j.name != pw_f.name and pw_j.name != pw_g.name and
                                    pw_j.name != pw_h.name and pw_j.name != pw_i.name and
                                    pw_g.implies(pw_j) and pw_h.implies(pw_j)):
                                    pw_chains.append((pw_f, pw_g, pw_h, pw_i, pw_j))
                                    if (len(pw_chains) >= subset_size): break
                        if (len(pw_chains) >= subset_size): break
                    if (len(pw_chains) >= subset_size): break

            else:  # De Morgan variant
                for pw_g, pw_h in combinations(pathways, 2):
                    neg_conj_gh = Pathway.conjunction(pw_g, pw_h).negation(self.all_genes)
                    pw_g_neg = pw_g.negation(self.all_genes)
                    pw_h_neg = pw_h.negation(self.all_genes)
                    for pw_f in pathways:
                        if (pw_f.name != pw_g.name and pw_f.name != pw_h.name and pw_f.implies(neg_conj_gh)):
                            pw_chains.extend([(pw_f, pw_g, pw_h, pw_j) for pw_j in pathways
                                              if (pw_j.name != pw_f.name and
                                                  pw_j.name != pw_g.name and
                                                  pw_j.name != pw_h.name and
                                                  pw_g_neg.implies(pw_j) and pw_h_neg.implies(pw_j))])
                            if (len(pw_chains) >= subset_size):
                                break
                    if (len(pw_chains) >= subset_size):
                        break

        return pw_chains[:subset_size]

    def get_genes_set(self, pw_chain: Tuple[Pathway, ...], scheme: SyllogisticScheme,
                      variant: SyllogisticSchemeVariant) -> Set[str]:
        if (variant in [SyllogisticSchemeVariant.BASE, SyllogisticSchemeVariant.NEGATION]):
            if (scheme == SyllogisticScheme.GEN_MODUS_PONENS):
                genes_set = pw_chain[0]
            else:  # Modus tollens
                if (variant == SyllogisticSchemeVariant.BASE):
                    genes_set = pw_chain[-1].negation(self.all_genes)
                else:
                    genes_set = pw_chain[-1]
        elif (variant == SyllogisticSchemeVariant.COMPLEX_PREDICATES):
            if (scheme == SyllogisticScheme.GEN_MODUS_PONENS):
                genes_set = Pathway.conjunction(pw_chain[0], pw_chain[2])
            else:  # Modus tollens
                genes_set = pw_chain[1].negation(self.all_genes)
        else:  # De Morgan variant
            if (scheme == SyllogisticScheme.GEN_MODUS_PONENS):
                genes_set = Pathway.conjunction(pw_chain[0].negation(self.all_genes),
                                                pw_chain[2].negation(self.all_genes))
            else:  # Modus tollens
                genes_set = Pathway.disjunction(pw_chain[1].negation(self.all_genes),
                                                pw_chain[2].negation(self.all_genes))

        return genes_set

    def get_premises(self, pw_chain: Tuple[Pathway, ...], scheme: SyllogisticScheme,
                     variant: SyllogisticSchemeVariant, gene_name: str = "") -> Dict[str, str]:
        premises = dict()
        transl = self.transl

        scheme_templ = self.scheme_templates[scheme][variant]
        if ("rel" in scheme_templ["premises"][0] or len(scheme_templ["premises"][0]["pw_chain"]) == 0):
            premises = {
                f"P{i + 1}": transl.translate("∀x:Ax→Bx", A=pw_chain[i].name, B=pw_chain[i + 1].name)
                for i in range(len(pw_chain) - 1)
            }

        for p_templ in scheme_templ["premises"]:
            params = {chr(ord("A") + i): pw_chain[arg_idx].name for i, arg_idx in enumerate(p_templ["pw_chain"])}
            if ("rel" in p_templ):
                p_idx = sorted(premises.keys())[p_templ["rel"]]
                premises[p_idx] = transl.translate(p_templ["formula"], **params)
            else:
                if (p_templ["pw_chain"]):
                    if ("a" in p_templ["formula"]):
                        params |= {"a": gene_name}
                    premises |= {f"P{len(premises) + 1}": transl.translate(p_templ["formula"], **params)}

        return premises

    def get_hypothesis(self, pw_chain: Tuple[Pathway, ...], scheme: SyllogisticScheme,
                       variant: SyllogisticSchemeVariant, gene_name: str = "") -> Dict[str, str]:
        transl = self.transl
        hyp_templ = self.scheme_templates[scheme][variant]["hypothesis"]
        params = {chr(ord("A") + i): pw_chain[arg_idx].name for i, arg_idx in enumerate(hyp_templ["pw_chain"])}
        if ("a" in hyp_templ["formula"]):
            params |= {"a": gene_name}

        hypothesis = transl.translate(hyp_templ["formula"], **params)

        return {"C": transl.translate("⊢", hyp=hypothesis)}


def get_ph_tuples(pwops: PathwaySetOps, scheme: SyllogisticScheme, variant: SyllogisticSchemeVariant,
                  length: int = 2, dummy: bool = False, subset_size: int = 200) -> List[Dict[str, str]]:
    ph_tuples = list()
    dummy_combs = combinations(ascii_uppercase, r=4)
    effective_length = length + 1 if ("modus" not in scheme) else length
    pw_chains = pwops.get_pw_chains(effective_length, scheme, variant, subset_size)
    for pw_chain in pw_chains:
        if (scheme in [SyllogisticScheme.GEN_MODUS_PONENS, SyllogisticScheme.GEN_MODUS_TOLLENS]):
            genes_set = pwops.get_genes_set(pw_chain, scheme, variant)
            for gene in list(genes_set)[:subset_size // 10]:
                gene_name = gene if not dummy else "".join(next(dummy_combs))
                premises = pwops.get_premises(pw_chain, scheme, variant, gene_name)
                hypothesis = pwops.get_hypothesis(pw_chain, scheme, variant, gene_name)
                ph_tuples.append(premises | hypothesis)
        else:
            premises = pwops.get_premises(pw_chain, scheme, variant)
            hypothesis = pwops.get_hypothesis(pw_chain, scheme, variant)
            ph_tuples.append(premises | hypothesis)

    return ph_tuples[:subset_size]


def falsify(ph_tuples: List[Dict[str, str]]):
    falsified_tuples = list()

    for ph_tuple in ph_tuples:
        falsified_tuples.append(ph_tuple.copy())
        falsified_tuples[-1]["C"] = falsified_tuples[-1]["C"].replace("It is true", "It is false")

    return falsified_tuples


def add_distractors(ph_tuple: Dict[str, str], pwops: PathwaySetOps, n: int):
    premise_keys = [pk for pk in ph_tuple if pk != "C"]
    genes = set()
    for p_key in premise_keys:
        if ("Gene" not in ph_tuple[p_key]):
            pathways = [pwops.get_pathway(pw) for pw in MEMBER_RGX.findall(ph_tuple[p_key])
                        if (pw != "Disease" and pw in pwops.tree)]
            genes.update(*pathways)

    unrel = [pw for pw in pwops.get_all_pathways() if (len(pw.intersection(genes)) == 0)]

    unrel_perms = list(permutations(unrel, 2))
    shuffle(unrel_perms)
    unrel_pairs = list()
    for pw1, pw2 in unrel_perms:
        if (pw1.implies(pw2)):
            unrel_pairs.append((pw1, pw2))
            if (len(unrel_pairs) >= n):
                break

    for i in range(n):
        ph_tuple[f"P{len(premise_keys) + 1 + i}"] = pwops.transl.translate("∀x:Ax→Bx", A=unrel_pairs[i][0].name,
                                                                           B=unrel_pairs[i][1].name)

    return ph_tuple


def add_distractors_all(ph_tuples: List[Dict[str, str]], pwops: PathwaySetOps, n: int = 5, seed: int = None):
    random.seed(seed)
    with Parallel(n_jobs=cpu_count(True)) as ppool:
        upd_ph_tuples = ppool(delayed(add_distractors)(ph_tuple, pwops, n)
                              for ph_tuple in tqdm(ph_tuples, desc="Adding distractors"))

    for i in range(len(ph_tuples)):
        for key in upd_ph_tuples[i]:
            ph_tuples[i][key] = upd_ph_tuples[i][key]

        # Puts hypothesis / conclusion back at the end of the dict structure.
        hyp = ph_tuples[i]["C"]
        del ph_tuples[i]["C"]
        ph_tuples[i]["C"] = hyp




