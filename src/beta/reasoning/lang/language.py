# Created by jing at 11.03.25
from itertools import product, combinations
import re

class Language:
    """
    A simple language for generating FOL clauses with necessary predicates.
    """

    def __init__(self):
        self.predicates = []

    def add_predicate(self, name, arity):
        """
        Add a predicate to the language.

        Args:
            name (str): The name of the predicate.
            arity (int): The number of arguments the predicate takes.
        """
        self.predicates[name] = arity

    def generate_candidate_clauses(self, symbolic_features):
        """
        Generates candidate clauses for individual predicates and combined predicates.

        Args:
            symbolic_features (list of lists or np.array): A matrix where each row is an object's features.

        Returns:
            list: A list of candidate FOL clause strings.
        """
        candidate_clauses = []

        # Generate clauses for each individual predicate.
        for predicate in self.predicates:
            clause = self._generate_clause_for_predicate(predicate, symbolic_features)
            if clause:
                candidate_clauses.append(clause)
        return candidate_clauses
    def merge_clauses(self, candidate_clauses):
        """
        Given a clause matrix of shape 4 x N (here N=11) where each row is a list of clause strings,
        merge the clauses column by column. For each column (i.e. each object), the clauses are merged
        into a single clause by:
          - Extracting the quantifier from the first clause (assumed to be identical across the column)
          - Removing the quantifier and trailing period from each clause
          - Joining the remaining predicate parts with " ∧ "
          - Prepending the quantifier and appending a period.

        Args:
            clause_matrix (list of lists): A list of 4 lists (one per predicate type), each with N clause strings.

        Returns:
            list: A list of N merged clause strings.
        """
        merged_clauses = []
        if not candidate_clauses or not candidate_clauses[0]:
            return merged_clauses

        num_columns = len(candidate_clauses[0])
        num_rows = len(candidate_clauses)

        for i in range(num_columns):
            # Extract the i-th clause from each row.
            col_clauses = [candidate_clauses[row][i] for row in range(num_rows)]

            # Use the first clause to determine the quantifier (e.g., "∃x" or "∀x").
            first_clause = col_clauses[0]
            m = re.match(r'^(∀x|∃x)\s+', first_clause)
            quantifier = m.group(1) if m else ""

            # Remove the quantifier and trailing period from each clause.
            predicate_parts = []
            for clause in col_clauses:
                # Remove quantifier.
                part = re.sub(r'^(∀x|∃x)\s+', '', clause)
                # Remove trailing period.
                part = part.rstrip('.')
                predicate_parts.append(part)

            # Merge the predicate parts with a logical AND.
            merged_body = " ∧ ".join(predicate_parts)
            # Prepend the quantifier and add a period.
            merged_clause = f"{quantifier} {merged_body}."
            merged_clauses.append(merged_clause)

        return merged_clauses


    def _generate_clause_for_predicate(self, predicate, symbolic_features):
        """
        Generates a clause for a single predicate based on its evaluation over all objects.
        """
        arity = predicate.arity
        # combos = list(product(symbolic_features, repeat=arity))
        clauses = predicate.evaluate(symbolic_features)
        return clauses
