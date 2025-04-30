# Created by MacBook Pro at 30.04.25

# clause_generation.py

from typing import List, Tuple, Set, NamedTuple


# A very simple Clause representation:
class Clause(NamedTuple):
    head: Tuple[str, ...]  # e.g. ("image_target", "X")
    body: Tuple[str, ...]  # e.g. ("has_shape", "o3", "triangle")


class ClauseGenerator:
    """
    Enumerate all single‐atom clauses from a collection of grounded hard‐facts.

    Given a list of training examples, each with:
      - 'hard_facts': List[Tuple[str, ...]]   e.g. ("has_shape", "o1", "triangle")
      - 'is_positive': bool
    this will collect every fact seen in at least one positive image,
    then emit for each such fact p(...) the candidate clause:

      image_target(X) :- p(...)
    """

    def __init__(self, target_pred: str = "image_target"):
        self.target_pred = target_pred

    def generate(self,
                 hard, soft
                 ) -> List[Clause]:
        """
        examples: List of dicts with keys
          - 'hard_facts': List[Tuple[str,...]]
          - 'is_positive': bool

        Returns: List of Clause
        """
        # 1) Collect every fact seen in ANY positive example
        pos_facts: Set[Tuple[str, ...]] = set()


        if ex.get("is_positive", False):
            for fact in ex["hard_facts"]:
                pos_facts.add(fact)

        # 2) Generate a candidate clause for each unique fact
        head = (self.target_pred, "X")
        clauses: List[Clause] = []
        for fact in sorted(pos_facts):
            # fact is a tuple like ("has_shape","o1","triangle")
            clauses.append(Clause(head=head, body=fact))

        return clauses

def generate_clauses(hard, soft):
    train_data = hard, soft
    cg = ClauseGenerator(target_pred="image_target")
    candidates = cg.generate(hard, soft)
    return candidates
