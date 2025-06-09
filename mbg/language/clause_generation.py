# Created by MacBook Pro at 30.04.25

# clause_generation.py

from typing import List, Tuple, Set, NamedTuple, Dict, Optional, Any, Union
from pathlib import Path
import torch
from collections import Counter
import json
import config
from mbg.grounding.predicates import HEAD_PREDICATES
from src import bk
import itertools
from dataclasses import dataclass

# A very simple Clause representation:
Atom = Tuple[Any, ...]  # e.g. ("has_shape", "o0", "triangle")
Head = Atom


class Clause:
    """
    A single candidate clause of the form:
        head :- body[0], body[1], ..., body[k-1].
    Optionally with a numeric weight.
    """

    def __init__(
            self,
            head: Head,
            body: List[Atom],
            weight: Optional[float] = None,
    ):
        self.head = head
        self.body = list(body)
        self.weight = weight

    def __repr__(self) -> str:
        # e.g. 'image_target(X) :- in_group(o0,g1), has_shape(o0,2), has_color(o0,(255,0,0)). [w=0.93]'
        body_str = ", ".join(
            f"{p}({','.join(map(str, args))})" if isinstance(p, str) else str((p, *args))
            for (p, *args) in self.body
        )
        head_str = f"{self.head[0]}({','.join(map(str, self.head[1:]))})"
        w = f" [w={self.weight:.3f}]" if self.weight is not None else ""
        return f"{head_str} :- {body_str}.{w}"

    def __eq__(self, other: Any) -> bool:
        return (
                isinstance(other, Clause)
                and self.head == other.head
                and set(self.body) == set(other.body)
                and (self.weight == other.weight)
        )

    def __hash__(self) -> int:
        # so we can dedupe in sets
        return hash((self.head, tuple(sorted(self.body)), self.weight))


# class ClauseGenerator:
#     """
#     Enumerate all single‐atom clauses (and some simple multi‐atom bodies)
#     from a collection of grounded hard‐ and soft‐facts.
#     """
#
#     def __init__(
#             self,
#             pos_pred: str = "image_target",
#             prox_thresh: float = 0.9,
#             sim_thresh: float = 0.5
#     ):
#         self.pos_pred = pos_pred
#         self.prox_thresh = prox_thresh
#         self.sim_thresh = sim_thresh
#
#     def generate(
#             self,
#             img_label: bool,
#             hard: Dict[str, torch.Tensor],
#             soft: Dict[str, torch.Tensor]
#     ) -> List[Clause]:
#         clauses: List[Clause] = []
#         head = (self.pos_pred, "X")
#
#         O = hard["has_shape"].size(0)
#         G = hard["group_size"].size(0)
#
#         # 1) single‐atom hard facts
#         for i in range(O):
#             oi = f"o{i}"
#             # shape
#             shape_val = int(hard["has_shape"][i].item())
#             clauses.append(Clause(head, [("has_shape", oi, shape_val)]))
#
#             # color
#             r, g, b = hard["has_color"][i].tolist()
#             clauses.append(Clause(head, [("has_color", oi, (int(r), int(g), int(b)))]))
#
#             # x, y, w, h
#             for coord in ("x", "y", "w", "h"):
#                 v = round(hard[coord][i].item(), 3)
#                 clauses.append(Clause(head, [(coord, oi, v)]))
#
#             # membership
#             for g in range(G):
#                 if hard["in_group"][i, g]:
#                     gi = f"g{g}"
#                     clauses.append(Clause(head, [("in_group", oi, gi)]))
#
#         # 2) group‐level single atoms
#         for g in range(G):
#             gi = f"g{g}"
#             sz = int(hard["group_size"][g].item())
#             pr = int(hard["principle"][g].item())
#             clauses.append(Clause(head, [("group_size", gi, sz)]))
#             clauses.append(Clause(head, [("principle", gi, pr)]))
#
#         # 3) soft facts above threshold
#         if "prox" in soft:
#             prox = soft["prox"]
#             for i in range(O):
#                 for j in range(i + 1, O):
#                     score = prox[i, j].item()
#                     if score >= self.prox_thresh:
#                         atom = ("prox", f"o{i}", f"o{j}")
#                         clauses.append(Clause(head, [atom], weight=score))
#
#         if "grp_sim" in soft:
#             grp_sim = soft["grp_sim"]
#             for g1 in range(G):
#                 for g2 in range(g1 + 1, G):
#                     score = grp_sim[g1, g2].item()
#                     if score >= self.sim_thresh:
#                         atom = ("grp_sim", f"g{g1}", f"g{g2}")
#                         clauses.append(Clause(head, [atom], weight=score))
#
#         # 4) multi‐atom bodies combining membership + two unary preds
#         unary_preds = ["has_shape", "has_color"]
#         for i in range(O):
#             oi = f"o{i}"
#             for g in range(G):
#                 if not hard["in_group"][i, g]:
#                     continue
#                 gi = f"g{g}"
#                 # collect values
#                 uni_vals = {}
#                 for p in unary_preds:
#                     t = hard[p][i]
#                     if t.dim() == 0:
#                         uni_vals[p] = int(t.item())
#                     else:
#                         uni_vals[p] = tuple(map(int, t.tolist()))
#
#                 for p1_idx in range(len(unary_preds)):
#                     for p2_idx in range(p1_idx + 1, len(unary_preds)):
#                         p1, p2 = unary_preds[p1_idx], unary_preds[p2_idx]
#                         v1, v2 = uni_vals[p1], uni_vals[p2]
#                         body = [
#                             ("in_group", oi, gi),
#                             (p1, oi, v1),
#                             (p2, oi, v2),
#                         ]
#                         clauses.append(Clause(head, body))
#
#         return clauses

class ClauseExtender:
    """
    Bottom-up clause extender using staged body-length growth.
    Takes an initial set of 1-body clauses (e.g. from image-level generator),
    evaluates them, and incrementally extends them to multi-body clauses.
    """

    def __init__(self, max_body_len=3, min_conf=0.8):
        self.max_body_len = max_body_len
        self.min_conf = min_conf

    def extend(self, pos_data, neg_data, fact_extractor) -> List[Clause]:
        all_atoms = set()
        pos_facts = []
        neg_facts = []

        for d in pos_data:
            facts = fact_extractor(d)
            pos_facts.append(facts)
            all_atoms.update(facts.keys())

        for d in neg_data:
            facts = fact_extractor(d)
            neg_facts.append(facts)
            all_atoms.update(facts.keys())

        body_clauses = [Clause(head=("image_target", "X"), body=[(a,)]) for a in sorted(all_atoms)]
        final_rules = []

        for L in range(1, self.max_body_len + 1):
            next_candidates = set()
            passed_clauses = []

            for clause in body_clauses:
                score = self._evaluate_clause(clause, pos_facts, neg_facts)
                if score >= self.min_conf:
                    clause.weight = score
                    final_rules.append(clause)
                    passed_clauses.append(clause)

            if not passed_clauses or L == self.max_body_len:
                break

            for c1, c2 in itertools.combinations(passed_clauses, 2):
                merged = sorted(set(c1.body) | set(c2.body))
                if len(merged) == L + 1:
                    next_candidates.add(Clause(head=("image_target", "X"), body=merged))

            body_clauses = list(next_candidates)

        return final_rules

    def _evaluate_clause(self, clause: Clause, pos_facts, neg_facts) -> float:
        pos_hits, neg_hits = 0, 0

        for facts in pos_facts:
            if all(facts.get(pred[0], False) for pred in clause.body):
                pos_hits += 1

        for facts in neg_facts:
            if all(facts.get(pred[0], False) for pred in clause.body):
                neg_hits += 1

        if pos_hits + neg_hits == 0:
            return 0.0
        return pos_hits / (pos_hits + neg_hits + 1e-5)


# class ClauseGenerator:
#     """
#     Enumerate single‐atom and simple multi‐atom clauses
#     from grounded hard/soft facts, _including duplicates_
#     so we can count how many times each clause fires.
#     """
#
#     def __init__(
#             self,
#             img_head: str = "image_target",
#             grp_head: str = "group_target",
#             prox_thresh: float = 0.9,
#             sim_thresh: float = 0.5
#     ):
#         self.img_head = img_head
#         self.grp_head = grp_head
#         self.prox_thresh = prox_thresh
#         self.sim_thresh = sim_thresh
#
#     def generate(
#             self,
#             hard: Dict[str, torch.Tensor],
#             soft: Dict[str, torch.Tensor]
#     ) -> List[Clause]:
#         clauses: List[Clause] = []
#
#         O = hard["has_shape"].size(0)
#         G = hard["group_size"].size(0)
#
#         def add(head, body, weight=None):
#             if weight is not None:
#                 clauses.append(Clause(head, body, weight=weight))
#             else:
#                 clauses.append(Clause(head, body))
#
#         # 1) image_target(X) clauses
#         img_head = (self.img_head, "X")
#
#         # (a) object-level shape-based predicates
#         for i in range(O):
#             shape_val = int(hard["has_shape"][i].item())
#
#             # has_shape
#             add(img_head, [("has_shape", "O", shape_val),
#                            ("in_group", "O", "G")])
#
#         # (a.2) group-level shape exclusion predicates
#         for pred in hard:
#             if pred.startswith("not_has_shape_"):
#                 shape_val = bk.bk_shapes.index(pred.split("not_has_shape_")[-1]) - 1
#                 if hard[pred].all():
#                     add(img_head, [(pred, "G", shape_val)])
#             if pred.startswith("diverse_") and pred!= "diverse_counts":
#                 if hard[pred].all():
#                     add(img_head, [(pred, "I")])
#             if pred.startswith("no_member_"):
#                 shape_val = bk.bk_shapes.index(pred.split("no_member_")[-1]) - 1
#                 if hard[pred].all():
#                     add(img_head, [(pred, "I", shape_val)])
#         # (b) one clause per object for color
#         for i in range(O):
#             rgb = tuple(int(c) for c in hard["has_color"][i].tolist())
#             add(img_head, [("has_color", "O", rgb),
#                            ("in_group", "O", "G")])
#         # (c) one clause per group for size & principle
#         for g in range(G):
#             sz = int(hard["group_size"][g].item())
#             pr = int(hard["principle"][g].item())
#             add(img_head, [("group_size", "G", sz)])
#             add(img_head, [("principle", "G", pr)])
#         # (d) soft object‐object proximity (one per qualifying pair)
#         if "prox" in soft:
#             prox = soft["prox"]
#             for i in range(O):
#                 for j in range(i + 1, O):
#                     score = prox[i, j].item()
#                     if score >= self.prox_thresh:
#                         add(img_head, [("prox", "O1", "O2")], weight=score)
#
#         # (e) soft group‐group similarity
#         if "grp_sim" in soft:
#             grp_sim = soft["grp_sim"]
#             for g1 in range(G):
#                 for g2 in range(g1 + 1, G):
#                     score = grp_sim[g1, g2].item()
#                     if score >= self.sim_thresh:
#                         add(img_head, [("grp_sim", "G1", "G2")], weight=score)
#
#         # 2) group_target(G,X) clauses
#         grp_head = (self.grp_head, "G", "X")
#
#         # (a) per object‐in‐that‐group shape
#         for i in range(O):
#             shape_val = int(hard["has_shape"][i].item())
#             for g in range(G):
#                 if hard["in_group"][i, g]:
#                     add(grp_head, [("has_shape", "O", shape_val),
#                                    ("in_group", "O", "G")])
#         # (a.2) group-level shape exclusion predicates
#         for pred in hard:
#             if pred.startswith("no_member_"):
#                 shape_val = bk.bk_shapes.index(pred.split("no_member_")[-1]) - 1
#                 for g in range(G):
#                     if hard[pred][g]:
#                         add(grp_head, [(pred, "G", shape_val)])
#         for pred in hard:
#             if pred.startswith("diverse_"):
#                 for g in range(G):
#                     if hard[pred][g]:
#                         add(grp_head, [(pred, "G", None)])
#             if pred.startswith("unique_"):
#                 for g in range(G):
#                     if hard[pred][g]:
#                         add(grp_head, [(pred, "G", None)])
#         # (b) per object‐in‐that‐group color
#         for i in range(O):
#             rgb = tuple(int(c) for c in hard["has_color"][i].tolist())
#             for g in range(G):
#                 if hard["in_group"][i, g]:
#                     add(grp_head, [("has_color", "O", rgb),
#                                    ("in_group", "O", "G")])
#
#         # (c) per‐group size/principle
#         for g in range(G):
#             sz = int(hard["group_size"][g].item())
#             pr = int(hard["principle"][g].item())
#             add(grp_head, [("group_size", "G", sz)])
#             add(grp_head, [("principle", "G", pr)])
#
#         # (f) symmetry-related object-object predicates
#         if all(k in hard for k in ["same_shape", "same_color", "mirror_x"]):
#             same_shape = hard["same_shape"]
#             same_color = hard["same_color"]
#             mirror = hard["mirror_x"]
#             for i in range(O):
#                 for j in range(i + 1, O):
#                     if mirror[i, j] > 0.5:
#                         if same_shape[i, j] > 0.5:
#                             add(img_head, [("mirror_x", "O1", "O2"),
#                                            ("same_shape", "O1", "O2")])
#                         if same_color[i, j] > 0.5:
#                             add(img_head, [("mirror_x", "O1", "O2"),
#                                            ("same_color", "O1", "O2")])
#         return clauses


@dataclass
class CWS:
    c: Clause
    support: torch.BoolTensor  # shape [O] or [G]


class ClauseGenerator:
    def __init__(self, img_head="image_target", grp_head="group_target", prox_thresh=0.9, sim_thresh=0.5):
        self.img_head = img_head
        self.grp_head = grp_head
        self.prox_thresh = prox_thresh
        self.sim_thresh = sim_thresh

    def _make_key(self, head, body, weight):
        # Use a tuple as hashable key (ignores weight for de-duplication if needed)
        return (head, tuple(body), round(weight, 4) if weight is not None else None)

    def generate(self, hard: Dict[str, torch.Tensor], soft: Dict[str, torch.Tensor]) -> List[CWS]:
        O = hard["has_shape"].size(0)
        G = hard["group_size"].size(0)

        clauses: List[CWS] = []
        seen_keys = set()

        clauses_dict: Dict[Tuple, CWS] = {}

        def add(head, body, weight=None, support_mask=None):
            key = self._make_key(head, body, weight)

            if key in clauses_dict:
                # OR the new support mask into the existing one
                existing = clauses_dict[key]
                if support_mask is not None and existing.support is not None:
                    combined = existing.support | support_mask
                else:
                    combined = support_mask or existing.support  # fallback
                clauses_dict[key] = CWS(existing.c, combined)
            else:
                clause = Clause(head, tuple(body), weight)
                clauses_dict[key] = CWS(clause, support_mask)

        img_head = (self.img_head, "X")
        grp_head = (self.grp_head, "G", "X")

        # (a) object-level shape
        for i in range(O):
            shape_val = int(hard["has_shape"][i].item())
            group_ids = torch.where(hard["in_group"][i])[0]
            for g in group_ids:
                mask = torch.zeros(O, dtype=torch.bool)
                mask[i] = True
                add(img_head, [("has_shape", "O", shape_val), ("in_group", "O", "G")], support_mask=mask)

        # (b) color
        for i in range(O):
            rgb = tuple(int(c) for c in hard["has_color"][i].tolist())
            group_ids = torch.where(hard["in_group"][i])[0]
            for g in group_ids:
                mask = torch.zeros(O, dtype=torch.bool)
                mask[i] = True
                add(img_head, [("has_color", "O", rgb), ("in_group", "O", "G")], support_mask=mask)

        # (c) group size & principle
        for g in range(G):
            sz = int(hard["group_size"][g].item())
            pr = int(hard["principle"][g].item())

            mask = torch.zeros(G, dtype=torch.bool)
            mask[g] = True
            add(img_head, [("group_size", "G", sz)], support_mask=mask)
            add(img_head, [("principle", "G", pr)], support_mask=mask)

        # (d) not_has_shape_*, no_member_*, diverse_*
        for pred in hard:
            if hard[pred].size() == 0:
                continue
            if pred.startswith("not_has_shape_"):

                shape_val = bk.bk_shapes.index(pred.split("not_has_shape_")[-1]) - 1
                if hard[pred].all():
                    add(img_head, [(pred, "G", shape_val)], support_mask=hard[pred].clone())

            elif pred.startswith("no_member_"):
                shape_val = bk.bk_shapes.index(pred.split("no_member_")[-1]) - 1
                mask = hard[pred].clone()
                add(img_head, [(pred, "I", shape_val)], support_mask=mask)

            elif pred.startswith("diverse_") and pred != "diverse_counts":

                mask = hard[pred].clone()
                add(img_head, [(pred, "I", None)], support_mask=mask)

        # (e) proximity
        if "prox" in soft:
            prox = soft["prox"]
            for i in range(O):
                for j in range(i + 1, O):
                    score = prox[i, j].item()
                    if score >= self.prox_thresh:
                        mask = torch.zeros(O, dtype=torch.bool)
                        mask[i] = True
                        mask[j] = True
                        add(img_head, [("prox", "O1", "O2")], weight=score, support_mask=mask)

        # (f) group-group similarity
        if "grp_sim" in soft:
            sim = soft["grp_sim"]
            for g1 in range(G):
                for g2 in range(g1 + 1, G):
                    score = sim[g1, g2].item()
                    if score >= self.sim_thresh:
                        mask = torch.zeros(G, dtype=torch.bool)
                        mask[g1] = True
                        mask[g2] = True
                        add(img_head, [("grp_sim", "G1", "G2")], weight=score, support_mask=mask)

        # (g) symmetry
        if all(k in hard for k in ["same_shape", "same_color", "mirror_x"]):
            mirror = hard["mirror_x"]
            same_shape = hard["same_shape"]
            same_color = hard["same_color"]
            for i in range(O):
                for j in range(i + 1, O):
                    if mirror[i, j] > 0.5:
                        mask = torch.zeros(O, dtype=torch.bool)
                        mask[i] = True
                        mask[j] = True
                        if same_shape[i, j] > 0.5:
                            add(img_head, [("mirror_x", "O1", "O2"), ("same_shape", "O1", "O2")], support_mask=mask)
                        if same_color[i, j] > 0.5:
                            add(img_head, [("mirror_x", "O1", "O2"), ("same_color", "O1", "O2")], support_mask=mask)

        # === Group-level clauses ===
        for g in range(G):

            g_mask = torch.zeros(G, dtype=torch.bool)
            g_mask[g] = True

            # (a) group_size, principle
            sz = int(hard["group_size"][g].item())
            pr = int(hard["principle"][g].item())
            add(grp_head, [("group_size", "G", sz)], support_mask=g_mask)
            add(grp_head, [("principle", "G", pr)], support_mask=g_mask)

            # (b) group-level predicates (diverse_*, unique_*)
            for pred in hard:
                if pred.startswith("diverse_") or pred.startswith("unique_"):
                    if hard[pred][g].item():
                        add(grp_head, [(pred, "G", None)], support_mask=g_mask)

            # (c) in-group object shape and color
            obj_ids = torch.where(hard["in_group"][:, g])[0]
            for i in obj_ids:
                shape_val = int(hard["has_shape"][i].item())
                rgb = tuple(int(c) for c in hard["has_color"][i].tolist())

                o_mask = torch.zeros(G, dtype=torch.bool)
                o_mask[g] = True

                add(grp_head, [("has_shape", "O", shape_val), ("in_group", "O", "G")], support_mask=o_mask)
                add(grp_head, [("has_color", "O", rgb), ("in_group", "O", "G")], support_mask=o_mask)

            # (d) no_member_* group-specific
            for pred in hard:
                if pred.startswith("no_member_"):
                    shape_val = bk.bk_shapes.index(pred.split("no_member_")[-1]) - 1
                    if hard[pred][g].item():
                        add(grp_head, [(pred, "G", shape_val)], support_mask=g_mask)
        return list(clauses_dict.values())


class ScoredRule(NamedTuple):
    c: Clause
    confidence: float
    scope: str  # one of "image", "existential", "universal"


def generate_clauses(
        img_label: bool,
        hard_facts: Dict[str, torch.Tensor],
        soft_facts: Dict[str, torch.Tensor],
        prox_thresh: float = 0.9,
        sim_thresh: float = 0.5,
) -> List[Clause]:
    """
    For a single example, produce all candidate clauses for *every*
    head predicate registered in HEAD_PREDICATES.

    Args:
      img_label:  whether this example is positive (unused by default)
      hard_facts: object‐ and group‐level hard tensors
      soft_facts: soft (neural) tensors
      prox_thresh: threshold for soft['prox']
      sim_thresh:  threshold for soft['grp_sim']

    Returns:
      A combined list of Clause(...) for each head in HEAD_PREDICATES.
    """

    gen = ClauseGenerator(
        prox_thresh=prox_thresh,
        sim_thresh=sim_thresh
    )
    # generate clauses with head_pred, e.g. image_target(X) or group_target(G,X)
    cls = gen.generate(
        hard=hard_facts,
        soft=soft_facts
    )
    freq = Counter(cls)

    return cls, freq


# def generate_clauses(img_label, hard, soft):
#     train_data = hard, soft
#     cg = ClauseGenerator(pos_pred="image_target")
#     candidates = cg.generate(img_label, hard, soft)
#     return candidates


def finalize_rules_per_task(pos_per_task, neg_per_task):
    """
    pos_per_task: Dict[task_id, List[List[Clause]]]
        each task_id → list over POSITIVE images → list of Clause for that image
    neg_per_task: same for NEGATIVE images
    output_dir:   Path where to write t{task_id}_rules.json
    """
    output_dir = config.output
    final_rules_per_task = {}
    for task_id, pos_lists in pos_per_task.items():
        neg_lists = neg_per_task.get(task_id, [])

        # 1) Count how many groups in each image via its 'group_size' atoms
        def count_groups(clause_list):
            return sum(
                1 for cl in clause_list
                if cl.head[0] == "group_target"
                and any(atom[0] == "group_size" for atom in cl.body)
            )

        pos_group_counts = [count_groups(pl) for pl in pos_lists]
        neg_group_counts = [count_groups(nl) for nl in neg_lists]

        # 2) Build per-image counters
        pos_counters = [Counter(pl) for pl in pos_lists]
        neg_counters = [Counter(nl) for nl in neg_lists]

        # 3) Collect every candidate clause seen in any positive image
        all_candidates = set().union(*pos_lists)

        kept = []
        for cl in all_candidates:
            head = cl.head[0]

            if head == "image_target":
                # must appear ≥1× in every POS image
                ok_pos = all(pc[cl] >= 1 for pc in pos_counters)
                # must appear 0× in every NEG image
                ok_neg = all(nc[cl] == 0 for nc in neg_counters)
                if ok_pos and ok_neg:
                    kept.append(cl)

            elif head == "group_target":
                # must appear exactly once per group in every POS image
                ok_pos = all(pc[cl] == gcount
                             for pc, gcount in zip(pos_counters, pos_group_counts))
                # in each NEG image, clause_count < #groups ⇒ at least one group fails ⇒ rule is false on that image
                if ok_pos:
                    print("")
                ok_neg = all(nc[cl] < gcount
                             for nc, gcount in zip(neg_counters, neg_group_counts))
                if ok_pos and ok_neg:
                    kept.append(cl)

        final_rules_per_task[task_id] = kept

        # write out JSON
        out_path = output_dir / f"t{task_id}_rules.json"
        with open(out_path, "w") as f:
            json.dump([cl.to_dict() for cl in kept], f, indent=2)
        print(f"[task {task_id}] discovered {len(kept)} rules → {out_path}")

    return final_rules_per_task


def split_clauses_by_head(
        pos_per_task: Dict[str, List[List[Clause]]],
        neg_per_task: Dict[str, List[List[Clause]]],
) -> Tuple[
    Dict[str, Counter],  # pos_image_counts[task] = Counter of image_target clauses
    Dict[str, List[Counter]],  # pos_group_counts[task] = list of Counters of group_target clauses per positive image
    Dict[str, Set[Clause]],  # neg_image_union[task]  = set of all image_target clauses seen in any negative image
    Dict[str, List[Counter]]  # neg_group_counts[task]  = list of Counters of group_target clauses per negative image
]:
    """
    Splits all candidate clauses by their head (image_target vs group_target).

    Args:
      pos_per_task: for each task_id, a list (over positive images) of List[Clause]
      neg_per_task: similarly for negative images

    Returns:
      pos_image_counts: how many positive images each image_target clause appeared in
      pos_group_counts: for each positive image, a Counter of its group_target clauses
      neg_image_union:  the union of all image_target clauses seen in negative images
      neg_group_counts: for each negative image, a Counter of its group_target clauses
    """
    IMG_HEAD = HEAD_PREDICATES["image"]  # e.g. "image_target"
    GRP_HEAD = HEAD_PREDICATES["group"]  # e.g. "group_target"

    pos_image_counts: Dict[str, Counter] = {}
    pos_group_counts: Dict[str, List[Counter]] = {}
    neg_image_union: Dict[str, Set[Clause]] = {}
    neg_group_counts: Dict[str, List[Counter]] = {}

    # --- positives ---
    for task_id, image_clauses in pos_per_task.items():
        img_ctr = Counter()
        grp_list: List[Counter] = []
        for clauses in image_clauses:
            this_grp_ctr = Counter()
            for cl in clauses:
                if cl.head[0] == IMG_HEAD:
                    img_ctr[cl] += 1
                elif cl.head[0] == GRP_HEAD:
                    this_grp_ctr[cl] += 1
            grp_list.append(this_grp_ctr)
        pos_image_counts[task_id] = img_ctr
        pos_group_counts[task_id] = grp_list

    # --- negatives ---
    for task_id, image_clauses in neg_per_task.items():
        img_set: Set[Clause] = set()
        grp_list: List[Counter] = []
        for clauses in image_clauses:
            this_grp_ctr = Counter()
            for cl in clauses:
                if cl.head[0] == IMG_HEAD:
                    img_set.add(cl)
                elif cl.head[0] == GRP_HEAD:
                    this_grp_ctr[cl] += 1
            grp_list.append(this_grp_ctr)
        neg_image_union[task_id] = img_set
        neg_group_counts[task_id] = grp_list

    return pos_image_counts, pos_group_counts, neg_image_union, neg_group_counts


def filter_image_level_rules(
        pos_freqs: List[List[CWS]],
        neg_freqs: List[List[CWS]],
) -> List[Tuple[Clause, float]]:
    """
    Score each clause based on its appearance across images:
      - support = (# positives with support) / N_pos
      - fpr     = (# negatives with support) / N_neg
      - score   = support * (1 - fpr)

    Returns:
      List of (clause, score), score in [0, 1]
    """

    from collections import defaultdict

    N_pos = len(pos_freqs)
    N_neg = len(neg_freqs)

    clause_pos_support = defaultdict(int)  # clause -> count of pos images with support
    clause_neg_support = defaultdict(int)  # clause -> count of neg images with support

    # Collect per-image presence for each clause in positives
    for freq_list in pos_freqs:
        present_in_image = set()
        for cws in freq_list:
            if cws.support.any().item():
                present_in_image.add(cws.c)
        for clause in present_in_image:
            clause_pos_support[clause] += 1

    # Collect per-image presence for each clause in negatives
    for freq_list in neg_freqs:
        present_in_image = set()
        for cws in freq_list:
            if cws.support.any().item():
                present_in_image.add(cws.c)
        for clause in present_in_image:
            clause_neg_support[clause] += 1

    # Union of all clauses
    all_clauses = set(clause_pos_support.keys()) | set(clause_neg_support.keys())

    scored: List[Tuple[Clause, float]] = []
    for clause in all_clauses:
        support = clause_pos_support[clause] / N_pos if N_pos > 0 else 0.0
        fpr = clause_neg_support[clause] / N_neg if N_neg > 0 else 0.0
        score = support * (1.0 - fpr)
        # if score>0.9:
        #     print("")
        scored.append((clause, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_group_existential_rules(
        pos_freqs: List[List[CWS]],
        neg_freqs: List[List[CWS]],
) -> List[Tuple[Clause, float]]:
    """
    For group_target clauses (group-level rules), score each clause r by:
      - support = (# positive images where r has ≥1 group match) / N_pos
      - fpr     = (# negative images where r has ≥1 group match) / N_neg
      - score   = support * (1 - fpr)

    Returns:
      List of (clause, score) sorted by score descending.
    """

    from collections import defaultdict

    N_pos = len(pos_freqs)
    N_neg = len(neg_freqs)

    pos_counts = defaultdict(int)  # clause -> count of positive images where it applies
    neg_counts = defaultdict(int)

    # Track group-level clauses: only those with head == 'group_target'
    def is_group_clause(clause):
        return clause.head[0] == "group_target"

    # Positive image counts
    for freq_list in pos_freqs:
        present = set()
        for cws in freq_list:
            if is_group_clause(cws.c) and cws.support.any().item():
                present.add(cws.c)
        for clause in present:
            pos_counts[clause] += 1

    # Negative image counts
    for freq_list in neg_freqs:
        present = set()
        for cws in freq_list:
            if is_group_clause(cws.c) and cws.support.any().item():
                present.add(cws.c)
        for clause in present:
            neg_counts[clause] += 1

    all_group_clauses = set(pos_counts.keys()) | set(neg_counts.keys())

    scored: List[Tuple[Clause, float]] = []
    for clause in all_group_clauses:
        support = pos_counts[clause] / N_pos if N_pos > 0 else 0.0
        fpr = neg_counts[clause] / N_neg if N_neg > 0 else 0.0
        score = support * (1.0 - fpr)
        if support > 0:
            scored.append((clause, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_group_universal_rules(
        pos_freqs: List[List[CWS]],
        neg_freqs: List[List[CWS]],
        pos_counts: List[int],
        neg_counts: List[int],
) -> List[Tuple[Clause, float]]:
    """
    Soft-universal filtering for group_target clauses using support masks.

    A clause is considered universally true in an image if it holds for all (or most) groups.
    We keep a clause only if it is NOT universally true in ANY negative image.

    For scoring:
      confidence = average of clipped group match ratio across positive images
      where ratio = (# matched groups) / (# total groups), clipped to ≤1
    """

    from collections import defaultdict

    # collect all candidate group_target clauses from positives
    candidates = set()
    for freq in pos_freqs:
        for cws in freq:
            if cws.c.head[0] == "group_target":
                candidates.add(cws.c)

    clause_to_pos_supports = defaultdict(list)
    clause_to_neg_supports = defaultdict(list)

    # collect support masks
    for img_idx, freq in enumerate(pos_freqs):
        for cws in freq:
            if cws.c.head[0] == "group_target":
                clause_to_pos_supports[cws.c].append(cws.support)

    for img_idx, freq in enumerate(neg_freqs):
        for cws in freq:
            if cws.c.head[0] == "group_target":
                clause_to_neg_supports[cws.c].append(cws.support)

    results = []
    for clause in candidates:
        pos_ratios = []
        neg_ratios = []
        # neg_universal = False

        # positive image ratios
        for i, support in enumerate(clause_to_pos_supports.get(clause, [])):
            total = len(support)
            if total > 0:
                ratio = support.sum().item() / total
                pos_ratios.append(min(ratio, 1.0))

        # negative image universal check
        for j, support in enumerate(clause_to_neg_supports.get(clause, [])):
            total = len(support)
            if total > 0:
                ratio = support.sum().item() / total
                neg_ratios.append(int(ratio == 1.0))

        if pos_ratios and neg_ratios:
            pos_support = sum(pos_ratios) / len(pos_freqs)
            neg_support = sum(neg_ratios) / len(neg_freqs) if neg_ratios else 0.0
            final_support = pos_support * (1.0 - neg_support)
            if final_support > 0:
                results.append((clause, final_support))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def assemble_final_rules(
        image_rules: List[Tuple[Clause, float]],
        exist_rules: List[Tuple[Clause, float]],
        universal_rules: List[Tuple[Clause, float]],
        top_k=5
) -> List[ScoredRule]:
    # for task_id in set(image_rules) | set(exist_rules) | set(universal_rules):
    all_scored = []

    for c, conf in image_rules:
        all_scored.append(ScoredRule(c, conf, "image"))

    for c, conf in exist_rules:
        all_scored.append(ScoredRule(c, conf, "existential"))

    for c, conf in universal_rules:
        all_scored.append(ScoredRule(c, conf, "universal"))

    # sort by confidence descending
    all_scored.sort(key=lambda sr: sr.confidence, reverse=True)
    return all_scored[:top_k]


def clause_to_text(clause: Clause) -> str:
    """
    Render a Clause as a human-readable string, e.g.
      image_target(X) :- has_shape(O,triangle), in_group(O,G)
    """
    head_pred, *head_args = clause.head
    head_args_str = ", ".join(head_args)
    head = f"{head_pred}({head_args_str})"
    if not clause.body:
        return head
    body_atoms = []
    for atom in clause.body:
        pred, *args = atom
        arg_str = ", ".join(map(str, args))
        body_atoms.append(f"{pred}({arg_str})")
    return head + " :- " + ", ".join(body_atoms)


def export_rules_to_json(
        final_rules: Dict[str, List[ScoredRule]],
        out_dir: Path
) -> None:
    """
    final_rules: mapping task_id -> list of ScoredRule
    out_dir: directory to write t<task_id>_rules.json
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for task_id, rules in final_rules.items():
        records: List[Dict[str, Any]] = []
        for r in rules:
            rec = {
                "text": clause_to_text(r.c),
                "confidence": float(r.score)
            }
            records.append(rec)
        path = out_dir / f"t{task_id}_rules.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"→ Wrote {len(records)} rules to {path}")
