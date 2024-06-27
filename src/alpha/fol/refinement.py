# Created by jing at 25.06.24
import numpy as np

import itertools
from .logic import Atom, Clause, Var


# TODOL refine_from_modeb, generate_by_refinement
class RefinementGenerator(object):
    """
    refinement operations for clause generation
    Parameters
    ----------
    lang : .language.Language
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, lang):
        self.lang = lang
        self.vi = 0  # counter for new variable generation

    def _check_recall(self, clause, mode_declaration):
        """Return a boolean value that represents the mode declaration can be used or not
        in terms of the recall.
        """
        return clause.count_by_predicate(mode_declaration.pred) < mode_declaration.recall
        # return self.recall_counter_dic[str(mode_declaration)] < mode_declaration.recall

    def get_max_obj_id(self, clause):
        vars = clause.all_vars_by_dtype('group')
        object_ids = [variable.id for variable in vars]
        object_names = [variable.name for variable in vars]
        if len(object_ids) == 0:
            return 0
        else:
            idx = np.argmax(object_ids)
            max_obj_id = np.max(object_ids)
            return max_obj_id, object_names[idx]

    def generate_new_variable(self, clause):
        obj_id, obj_name = self.get_max_obj_id(clause)
        obj_name = str(obj_name).split("_")[0]
        return Var(obj_name + "_" + str(obj_id + 1))

    def refine_from_modeb(self, clause, modeb):
        """Generate clauses by adding atoms to body using mode declaration.

        Args:
              clause (Clause): A clause.
              modeb (ModeDeclaration): A mode declaration for body.
        """
        # list(list(Term))
        if not self._check_recall(clause, modeb):
            # the input modeb has been used as many as its recall (maximum number  to be called) already
            return []
        # unused_args = logic_utils.get_clause_unused_args(clause)
        terms_list = self.generate_term_combinations(clause, modeb)
        C_refined = []
        for terms in terms_list:
            if len(terms) == len(list(set(terms))):
                if not modeb.ordered:
                    terms = sorted(terms)
                new_atom = Atom(modeb.pred, terms)
                if not new_atom in clause.body:
                    body = sorted(clause.body) + sorted([new_atom])
                    new_clause = Clause(clause.head, body)
                    C_refined.append(new_clause)
        return list(set(C_refined))

    def generate_term_combinations(self, clause, modeb):
        """Generate possible term list for new body atom.
        Enumerate possible assignments for each place in the mode predicate,
        generate all possible assignments by enumerating the combinations.

        Args:
            modeb (ModeDeclaration): A mode declaration for body.
        """

        assignments_list = []
        term_num = 0
        for mt in modeb.mode_terms:
            if mt.dtype.name == "group":
                term_num += 1
        for mt in modeb.mode_terms:
            assignments = []
            if mt.mode == '+':
                # var_candidates = clause.var_all()
                assignments = clause.all_vars_by_dtype(mt.dtype)
            elif mt.mode == '-':
                # get new variable
                # How to think as candidates? maybe [O3] etc.
                # we get only object variable e.g. O3
                # new_var = self.generate_new_variable()
                assignments = [self.generate_new_variable(clause)]
            elif mt.mode == '#':
                # consts = self.lang.get_by_dtype(mt.mode.dtype)
                assignments = self.lang.get_by_dtype(mt.dtype)

            assignments_list.append(assignments)
        # generate all combinations by cartesian product
        # e.g. [[O2], [red,blue,yellow]]
        # -> [[O2,red],[O2,blue],[O2,yellow]]
        ##print(assignments_list)
        ##print(list(itertools.product(*assignments_list)))
        ##print(clause, modeb, assignments_list)
        # print(clause, modeb)
        # print(assignments_list)
        if modeb.ordered:
            return itertools.product(*assignments_list)
        else:
            arg_lists = []
            if len(assignments_list) == 5:
                for i_1, a_1 in enumerate(assignments_list[0]):
                    for i_2 in range(i_1 + 1, len(assignments_list[1])):
                        for i_3 in range(i_2 + 1, len(assignments_list[2])):
                            for i_4 in range(i_3 + 1, len(assignments_list[3])):
                                for a_5 in assignments_list[4]:
                                    arg_lists.append([assignments_list[0][i_1],
                                                      assignments_list[1][i_2],
                                                      assignments_list[2][i_3],
                                                      assignments_list[3][i_4], a_5])
            if len(assignments_list) == 4:
                for i_1, a_1 in enumerate(assignments_list[0]):
                    for i_2 in range(i_1 + 1, len(assignments_list[1])):
                        for i_3 in range(i_2 + 1, len(assignments_list[2])):
                            for a_4 in assignments_list[3]:
                                arg_lists.append([a_1, assignments_list[1][i_2], assignments_list[2][i_3], a_4])
            elif len(assignments_list) == 3:
                for i_1, a_1 in enumerate(assignments_list[0]):
                    for a_2 in assignments_list[1][i_1 + 1:]:
                        for i_3, a_3 in enumerate(assignments_list[2]):
                            arg_lists.append([a_1, a_2, a_3])
            elif len(assignments_list) == 2:
                for i_1, a_1 in enumerate(assignments_list[0]):
                    for a_2 in assignments_list[1]:
                        arg_lists.append([a_1, a_2])
            elif len(assignments_list) == 1:
                for a in assignments_list[0]:
                    arg_lists.append([a])

            return arg_lists

    def refinement_clause(self, clause):
        C_refined = []
        for modeb in self.lang.mode_declarations:
            new_clauses = self.refine_from_modeb(clause, modeb)
            # new_clauses = [c for c in new_clauses if self._is_valid(c)]
            C_refined.extend(new_clauses)
            ##print(C_refined)
        return list(set(C_refined))
