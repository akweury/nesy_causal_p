# Created by MacBook Pro at 30.04.25

# clause_generation.py

from typing import List, Tuple, Set, NamedTuple, Dict, Optional, Any
import torch
from collections import Counter, defaultdict, namedtuple
import json
import config
from mbg.grounding.predicates import HEAD_PREDICATES, OBJ_HARD, GRP_HARD, SOFT
from torch import Tensor

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



class ClauseGenerator:
    """
    Enumerate single‐atom and simple multi‐atom clauses
    from grounded hard/soft facts, _including duplicates_
    so we can count how many times each clause fires.
    """

    def __init__(
        self,
        img_head: str = "image_target",
        grp_head: str = "group_target",
        prox_thresh: float = 0.9,
        sim_thresh: float = 0.5
    ):
        self.img_head   = img_head
        self.grp_head   = grp_head
        self.prox_thresh = prox_thresh
        self.sim_thresh  = sim_thresh

    def generate(
        self,
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor]
    ) -> List[Clause]:
        clauses: List[Clause] = []

        O = hard["has_shape"].size(0)
        G = hard["group_size"].size(0)

        def add(head, body, weight=None):
            if weight is not None:
                clauses.append(Clause(head, body, weight=weight))
            else:
                clauses.append(Clause(head, body))

        # 1) image_target(X) clauses
        img_head = (self.img_head, "X")

        # (a) one clause per object for shape
        for i in range(O):
            shape_val = int(hard["has_shape"][i].item())
            add(img_head, [("has_shape", "O", shape_val),
                           ("in_group",  "O", "G")])

        # (b) one clause per object for color
        for i in range(O):
            rgb = tuple(int(c) for c in hard["has_color"][i].tolist())
            add(img_head, [("has_color", "O", rgb),
                           ("in_group",  "O", "G")])

        # (c) one clause per group for size & principle
        for g in range(G):
            sz = int(hard["group_size"][g].item())
            pr = int(hard["principle"][g].item())
            add(img_head, [("group_size",  "G", sz)])
            add(img_head, [("principle",   "G", pr)])

        # (d) soft object‐object proximity (one per qualifying pair)
        if "prox" in soft:
            prox = soft["prox"]
            for i in range(O):
                for j in range(i+1, O):
                    score = prox[i,j].item()
                    if score >= self.prox_thresh:
                        add(img_head, [("prox", "O1", "O2")], weight=score)

        # (e) soft group‐group similarity
        if "grp_sim" in soft:
            grp_sim = soft["grp_sim"]
            for g1 in range(G):
                for g2 in range(g1+1, G):
                    score = grp_sim[g1,g2].item()
                    if score >= self.sim_thresh:
                        add(img_head, [("grp_sim", "G1", "G2")], weight=score)


        # 2) group_target(G,X) clauses
        grp_head = (self.grp_head, "G", "X")

        # (a) per object‐in‐that‐group shape
        for i in range(O):
            shape_val = int(hard["has_shape"][i].item())
            # membership matrix tells us which groups this object belongs to
            for g in range(G):
                if hard["in_group"][i, g]:
                    add(grp_head, [("has_shape", "O", shape_val),
                                   ("in_group",  "O", "G")])

        # (b) per object‐in‐that‐group color
        for i in range(O):
            rgb = tuple(int(c) for c in hard["has_color"][i].tolist())
            for g in range(G):
                if hard["in_group"][i, g]:
                    add(grp_head, [("has_color", "O", rgb),
                                   ("in_group",  "O", "G")])

        # (c) per‐group size/principle
        for g in range(G):
            sz = int(hard["group_size"][g].item())
            pr = int(hard["principle"][g].item())
            add(grp_head, [("group_size",  "G", sz)])
            add(grp_head, [("principle",   "G", pr)])

        return clauses


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
    pos_per_task: Dict[str, List[Counter]],
    neg_per_task: Dict[str, List[Counter]],
) -> Dict[str, List[Clause]]:
    """
    Keep only those image‐head clauses that:
      - appear in every positive image (freq>0 in each Counter)
      - appear in no negative image (freq=0 in all neg Counters)
    """
    final: Dict[str, List[Clause]] = {}
    for task_id, pos_freqs in pos_per_task.items():
        neg_freqs = neg_per_task.get(task_id, [])
        # 1) start from intersection across all positives
        common = set(pos_freqs[0].keys())
        for freq in pos_freqs[1:]:
            common &= set(freq.keys())
        # 2) subtract any that appears in any negative
        for freq in neg_freqs:
            common -= set(freq.keys())
        final[task_id] = list(common)
    return final

def filter_group_existential_rules(
    pos_per_task: Dict[str, List[Counter]],
    neg_per_task: Dict[str, List[Counter]],
) -> Dict[str, List[Clause]]:
    """
    For group‐target clauses only, keep those that:
      - for every positive image, clause appears at least once (exists a group)
      - for every negative image, clause appears zero times (no group)
    """
    final: Dict[str, List[Clause]] = {}
    for task_id, pos_freqs in pos_per_task.items():
        neg_freqs = neg_per_task.get(task_id, [])
        # extract only group_target clauses
        def is_grp(c: Clause) -> bool:
            return c.head[0] == "group_target"

        # intersect across pos
        common = set(filter(is_grp, pos_freqs[0].keys()))
        for freq in pos_freqs[1:]:
            common &= {c for c in freq if is_grp(c)}

        # subtract any that appears in any negative
        for freq in neg_freqs:
            common -= {c for c in freq if is_grp(c)}

        final[task_id] = list(common)
    return final

def filter_group_universal_rules(
    pos_per_task: Dict[str, List[Counter]],
    neg_per_task: Dict[str, List[Counter]],
    pos_group_counts: Dict[str, List[int]],
    neg_group_counts: Dict[str, List[int]],
) -> Dict[str, List[Clause]]:
    """
    For group‐target clauses only, keep those that:
      - in every positive image i, clause count == total_groups[i]  (holds for *all* groups)
      - in no negative image j does clause count == total_groups[j] (never holds for *all* groups)
    """
    final: Dict[str, List[Clause]] = {}
    for task_id, pos_freqs in pos_per_task.items():
        neg_freqs = neg_per_task.get(task_id, [])
        pos_counts = pos_group_counts[task_id]
        neg_counts = neg_group_counts.get(task_id, [])

        # collect all candidate group‐target clauses seen in any pos
        candidates = set()
        for freq in pos_freqs:
            candidates |= {c for c in freq if c.head[0] == "group_target"}

        keep: List[Clause] = []
        for c in candidates:
            # 1) must hold in *all* groups of each positive image
            ok_pos = all(freq.get(c, 0) == pos_counts[i]
                         for i, freq in enumerate(pos_freqs))
            if not ok_pos:
                continue
            # 2) must *not* hold in *all* groups in *any* negative image
            ok_neg = all(freq.get(c, 0) < neg_counts[j]
                         for j, freq in enumerate(neg_freqs))
            if ok_neg:
                keep.append(c)

        final[task_id] = keep
    return final

