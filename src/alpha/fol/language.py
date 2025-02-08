# Created by shaji at 24/06/2024

from lark import Lark
import itertools
import torch.nn.functional as F

from src.alpha.fol.exp_parser import ExpTree
from src.alpha.fol.logic import *
from src.alpha.fol import lang_utils, mode_declaration
from src import bk


def get_unused_args(c):
    unused_args = []
    used_args = []
    for body in c.body:
        if "has" in body.pred.name:
            unused_args.append(body.terms[0])
    for body in c.body:
        if "has" not in body.pred.name:
            for term in body.terms:
                term_name = term.name.split("_")[0]
                if term_name == bk.variable["feature_map"] and term not in used_args:
                    unused_args.remove(term)
                    used_args.append(term)
    return unused_args, used_args


def sub_lists(lst):
    # Generate all non-empty sublists
    sublists = [lst[i:j] for i in range(len(lst)) for j in
                range(i + 1, len(lst) + 1)]
    return sublists


class Language(object):
    """Language of first-order logic.

    A class of languages in first-order logic.

    Args:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.

    Attrs:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.
    """

    def __init__(self, obj_num, variable_group_symbol, variable_obj_symbol,
                 lark_path, phi_num, rho_num, number):
        self.obj_vars = [Var(f"{variable_obj_symbol}_{v_i}", bk.var_dtypes["object"])
                         for v_i in range(obj_num)]
        self.obj_variable_num = obj_num
        self.variable_group_symbol = variable_group_symbol
        self.phi_num = phi_num
        self.rho_num = rho_num
        self.number = number
        self.atoms = []
        self.funcs = []
        self.consts = []
        self.predicates = []
        self.mode_declarations = None
        # PI
        self.invented_preds = []
        self.inv_p_with_scores = []
        self.all_inv_p = []
        self.pi_c = []
        self.clauses = []
        self.clause_with_scores = []
        self.invented_predicates_number = 0
        self.invented_consts_number = 0

        # Results
        self.record_clauses = []
        self.record_atoms = []
        self.record_consts = []
        self.record_predicates = []
        self.record_final_clauses = []
        self.record_group_variable_num = 0

        # GRAMMAR
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")

        # load BK predicates and constants
        self.obj_predicates = self.load_preds("object")
        self.group_predicates = self.load_preds("group")

    def load_preds(self, level):
        # load all the bk predicates
        predicates = []
        if level == "object":
            for pred_config in bk.predicate_configs.values():
                predicates.append(self.parse_pred(pred_config))
        elif level =="group":
            for pred_config in bk.group_predicate_configs.values():
                predicates.append(self.parse_pred(pred_config))
        else:
            raise Exception("Invalid predicate level")
        return predicates

    def reset_lang(self, g_num, level):
        self.done = False
        self.consts, self.min_consts = self.load_consts(self.number, g_num, self.phi_num,
                                                        self.rho_num, 1)
        self.group_vars = [
            Var(f"{self.variable_group_symbol}_{v_i}", bk.var_dtypes["group"]) for
            v_i in range(g_num)]
        self.group_variable_num = g_num
        # update predicates
        self.predicates = self.load_preds(level)
        self.clauses = []
        self.generate_atoms()
        # update language
        self.mode_declarations = mode_declaration.get_mode_declarations(
            self.predicates)

    def __str__(self):
        s = "===Predicates===\n"
        for pred in self.predicates:
            s += pred.__str__() + '\n'
        s += "===Function Symbols===\n"
        for func in self.funcs:
            s += func.__str__() + '\n'
        s += "===Constants===\n"
        for const in self.consts:
            s += const.__str__() + '\n'
        s += "===Invented Predicates===\n"
        for invented_predicates in self.invented_preds:
            s += invented_predicates.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def update_consts(self, clauses):
        new_consts = self.min_consts
        occurred_consts = []
        for c in clauses:
            for atom in c.body:
                for term in atom.terms:
                    if isinstance(term, Const) and term not in new_consts:
                        new_consts.append(term)
                        occurred_consts.append(term)
        self.consts = new_consts
        self.occurred_consts = occurred_consts

    def generate_minimum_atoms(self, prim_args_list):
        p_ = Predicate('.', 1, [mode_declaration.DataType('spec')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype, with_inv=True) for dtype in
                           dtypes]
            if pred.pi_type == "clu_pred":
                consts_list = [[atom.terms[0]] for atom in pred.body[0]]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 4:
                    if args[:3] in prim_args_list:
                        atoms.append(Atom(pred, args))

                elif len(args) == 1 or len(set(args)) == len(args):
                    atoms.append(Atom(pred, args))
        pi_atoms = []
        for pred in self.invented_preds:
            consts_list = []
            for body_pred in pred.body[0]:
                consts_list.append([body_pred.terms[0]])

            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    new_atom = Atom(pred, args)
                    if new_atom not in atoms:
                        pi_atoms.append(new_atom)
        bk_pi_atoms = []
        for pred in self.bk_inv_preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype, with_inv=True) for dtype in
                           dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                # check if args and pred correspond are in the same area
                if pred.dtypes[0].name == 'area':
                    if pred.name[0] + pred.name[5:] != args[0].name:
                        continue
                if len(args) == 1 or len(set(args)) == len(args):
                    pi_atoms.append(Atom(pred, args))
        self.atoms = spec_atoms + sorted(atoms) + sorted(bk_pi_atoms) + sorted(
            pi_atoms)

    def unique_combinations_filter(self, list_of_lists):
        """
        Given a list of lists, return all combinations where one element is chosen from each list,
        while discarding:
          - Combinations that contain repeated elements.
          - Combinations that are duplicates up to ordering (i.e. same set of elements).

        Args:
            list_of_lists (list of lists): Input lists.

        Returns:
            list of tuples: Unique combinations (one element per list) meeting the above criteria.
        """
        seen = set()  # to track canonical forms (sorted tuples)
        valid_combos = []  # to store the valid combinations

        for combo in itertools.product(*list_of_lists):
            # Discard if any element is repeated.
            if len(set(combo)) != len(combo):
                continue

            # Create a canonical representation by sorting the tuple.
            # This representation will be the same for combinations with the same elements.
            canonical = tuple(sorted(combo))

            if canonical not in seen:
                seen.add(canonical)
                valid_combos.append(combo)

        return valid_combos
    def generate_atoms(self, clauses=None):
        p_ = Predicate('.', 1, [DataType('spec,?')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec,?'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec,?'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.predicates:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            args_list = self.unique_combinations_filter(consts_list)
            # # Generate all possible combinations (Cartesian product)
            # args_list = itertools.product(*consts_list)
            # # Filter out combinations that contain duplicate elements
            # args_list = list(set([combo for combo in args_list if len(set(combo)) == len(combo)]))
            #
            # args_list = list(set(itertools.product(*consts_list)))
            if isinstance(pred, NeuralPredicate):
                for args in args_list:
                    atoms.append(Atom(pred, args))
            elif isinstance(pred, InventedPredicate):
                dtypes = pred.dtypes
                consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
                all_combs = list(itertools.product(*consts_list))

                # args_list = self.unique_combinations_filter(consts_list)
                for args in all_combs:
                    atoms.append(InvAtom(pred, args))
        if clauses is not None:
            for clause in clauses:
                if clause.body[0] not in atoms:
                    atoms.append(clause.body[0])
        self.atoms = spec_atoms + sorted(atoms)

    def assign_terms(self, mode, dtype, vars):
        if mode == "#":
            return self.get_by_dtype(dtype)
        elif mode == "+":
            # TODO: pattern and group are treated separately
            dtype_vars = [v for v in vars if v.var_type == dtype]
            return dtype_vars
        else:
            raise ValueError

    def generate_inv_atoms(self, pred, vars):
        # return ungrounded atoms
        # dtype_list = pred.arg_list

        term_list = []
        for sub_pred in pred.sub_preds:
            dtypes = sub_pred.dtypes
            const_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            grounded_terms = list(set(itertools.product(*const_list)))
            # grounded_terms = [list(t) for t in grounded_terms]
            term_list.append(grounded_terms)
        term_list = list(set(itertools.product(*term_list)))
        term_list = lang_utils.remove_chaos_terms(term_list)
        term_list = tuple([tuple(lang_utils.orgnize_inv_pred_dtypes(terms)) for terms in term_list])

        grounded_atoms = []
        for terms in term_list:
            grounded_atoms.append(InvAtom(pred, terms))

        ungrounded_term_list = []
        for sub_pred in pred.sub_preds:
            dtypes = sub_pred.dtypes
            assignment_list = [self.assign_terms(dtype.data[1], dtype, vars) for dtype in dtypes]
            ungrounded_terms = list(itertools.product(*assignment_list))
            ungrounded_term_list.append(ungrounded_terms)
        ungrounded_term_list = list(set(itertools.product(*ungrounded_term_list)))
        ungrounded_term_list = [tuple(lang_utils.orgnize_inv_pred_dtypes(terms)) for terms in ungrounded_term_list]
        ungrounded_atoms = []
        for terms in ungrounded_term_list:
            ungrounded_atoms.append(InvAtom(pred, terms))

        return grounded_atoms, ungrounded_atoms

    def load_obj_init_clauses(self):
        """Read lines and parse to Atom objects.
        """
        obj_clauses_str = []
        var_pattern = bk.variable['pattern']
        pred_target = bk.predicate_configs["predicate_target"].split(':')[0]
        pred_pattern_in = bk.predicate_configs["predicate_in_pattern"].split(':')[0]
        pred_group_in = bk.predicate_configs["predicate_in_group"].split(':')[0]

        head = f"{pred_target}({var_pattern}):-"
        body = ""
        body += f"{pred_pattern_in}({self.group_vars[0]},{var_pattern}),"
        for o_i in range(self.obj_variable_num):
            body += f"{pred_group_in}({self.obj_vars[o_i]},{self.group_vars[0]},{var_pattern}),"
        obj_clauses_str.append(head + body[:-1] + ".")
        obj_clauses = []
        for group_clause_str in obj_clauses_str:
            tree = self.lp_clause.parse(group_clause_str)
            group_clause = ExpTree(self).transform(tree)
            obj_clauses.append(group_clause)
        return obj_clauses

    def load_group_init_clauses(self):
        """Read lines and parse to Atom objects.
        """
        group_clauses_str = []
        var_pattern = bk.variable['pattern']
        pred_target = bk.predicate_configs["predicate_target"].split(':')[0]
        pred_pattern_in = bk.predicate_configs["predicate_in_pattern"].split(':')[0]

        head = f"{pred_target}({var_pattern}):-"
        body = ""
        for var in self.group_vars:
            body += f"{pred_pattern_in}({var},{var_pattern}),"
        # for var in self.group_vars:
        #     body += f"{pred_pattern_in}({var},{var_pattern}),"
        group_clauses_str.append(head + body[:-1] + ".")
        group_clauses = []
        for group_clause_str in group_clauses_str:
            tree = self.lp_clause.parse(group_clause_str)
            group_clause = ExpTree(self).transform(tree)
            group_clauses.append(group_clause)
        return group_clauses

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        head_str, arity, dtype_names_str = line.split(':')
        dtype_data = dtype_names_str.split(';')
        dtypes = [mode_declaration.DataType(dt) for dt in dtype_data]
        return NeuralPredicate(head_str, int(arity), dtypes)

    def parse_min_const(self, number, g_num, obj_num, const, const_type):
        """Parse string to function symbols.
        """
        const_data_type = mode_declaration.DataType(const)
        const_names = []
        if "amount_" in const_type:
            _, num = const_type.split('_')
            if num == "group":
                num = g_num
            elif num == "object":
                num = obj_num
            const_names = []
            for i in range(int(num)):
                const_names.append(f"{const_data_type.name}{i + 1}of{num}")
        elif 'pattern' == const_type:
            const_names = ['data']
        elif 'group_pattern' == const_type:
            const_names = ['group']
        consts = []
        for c_i, name in enumerate(const_names):
            consts.append(Const(name, const_data_type))
        return consts

    def parse_const(self, number, g_num, phi_num, rho_num, obj_num, const, const_type):
        """Parse string to function symbols.
        """
        const_data_type = mode_declaration.DataType(const)
        if "amount_" in const_type:
            _, num = const_type.split('_')
            if num == "group":
                num = g_num
            elif num == "object":
                num = obj_num
            const_names = []
            for i in range(int(num)):
                const_names.append(f"{const_data_type.name}{i + 1}of{num}")
        elif 'pattern' == const_type:
            const_names = ['data']
        elif 'group_pattern' == const_type:
            const_names = ['group']
        elif "quantity" == const_type:
            const_names = []
            for i in range(int(number)):
                const_names.append(f"{const_data_type.name}{i + 1}")
        elif 'enum' in const_type:
            if const_data_type.name == bk.const_dtypes["object_color"]:
                const_names = bk.color_large
            elif const_data_type.name == bk.const_dtypes["object_shape"]:
                const_names = bk.bk_shapes
            elif const_data_type.name == bk.const_dtypes["group_label"]:
                const_names = bk.bk_shapes
            else:
                raise ValueError
        else:
            raise ValueError
        consts = []
        for c_i, name in enumerate(const_names):
            const = Const(name, const_data_type)
            consts.append(const)
        return consts

    def load_consts(self, number, g_num, phi_num, rho_num, obj_num):
        consts = []
        min_consts = []
        for const_name, const_type in bk.const_dict.items():
            min_consts.extend(
                self.parse_min_const(number, g_num, obj_num, const_name, const_type))
            consts.extend(
                self.parse_const(number, g_num, phi_num, rho_num, obj_num, const_name, const_type))
        return consts, min_consts

    def rename_bk_preds_in_clause(self, bk_prefix, line):
        """Parse string to invented predicates.
        """
        new_line = line.replace('\n', '')
        new_line = new_line.replace('inv_pred', "inv_pred_bk" + str(bk_prefix) + "_")
        return new_line

    def parse_invented_bk_clause(self, line, lang):
        """Parse string to invented predicates.
        """

        tree = self.lp_clause.parse(line)
        clause = ExpTree(lang).transform(tree)

        return clause

    def parse_invented_bk_pred(self, line):
        """Parse string to invented predicates.
        """
        head, body = line.split(':-')
        arity = len(head.split(","))
        head_dtype_names = arity * ['group']
        dtypes = [mode_declaration.DataType(dt) for dt in head_dtype_names]

        # pred_with_id = pred + f"_{i}"
        pred_with_id = head.split("(")[0]
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes,
                                          args=None, pi_type=None)

        return invented_pred

    def count_arity_from_clauses(self, clause_cluster):
        OX = "O"
        arity = 1
        arity_list = []
        id = clause_cluster[0].split("inv_pred")[1].split("(")[0]
        while (OX + str(arity)) in clause_cluster[0]:
            arity_list.append(OX + str(arity))
            arity += 1
        return arity_list, id

    def load_invented_preds(self, clauses_str, target_clauses_str):
        inv_p_clauses = []
        inv_preds = []
        # generate clauses
        for clause_str in clauses_str:
            inv_pred = self.parse_invented_bk_pred(clause_str)
            if inv_pred not in self.invented_preds:
                self.invented_preds.append(inv_pred)
            inv_preds.append(inv_pred)
            tree = self.lp_clause.parse(clause_str)
            clause = ExpTree(self).transform(tree)
            # generate clauses
            inv_p_clauses.append(clause)

        self.all_invented_preds = self.invented_preds
        self.all_pi_clauses = inv_p_clauses

        target_clauses = []
        for target_clause_str in target_clauses_str:
            target_clause_str = target_clause_str.replace(" ", "")
            tree = self.lp_clause.parse(target_clause_str)
            clause = ExpTree(self).transform(tree)
            # generate clauses
            target_clauses.append(clause)
        self.all_clauses = target_clauses
        # unique predicate
        new_predicates = []
        p_names = []
        for pred in inv_preds:
            if "inv" in pred.name and pred.name not in p_names:
                p_names.append(pred.name)
                new_predicates.append(pred)

        for inv_pred in self.invented_preds:
            inv_pred.body = []
            for c in inv_p_clauses:
                if c.head.pred.name == inv_pred.name:
                    inv_pred.body.append(c.body)

        self.update_inv()

    def update_inv(self):
        self.invented_preds = self.all_invented_preds
        self.pi_clauses = self.all_pi_clauses
        self.generate_atoms()

    def get_var_and_dtype(self, atom):
        """Get all variables in an input atom with its dtypes by enumerating variables in the input atom.

        Note:
            with the assumption with function free atoms.

        Args:
            atom (Atom): The atom.

        Returns:
            List of tuples (var, dtype)
        """
        var_dtype_list = []
        for i, arg in enumerate(atom.terms):
            if arg.is_var():
                dtype = atom.pred.dtypes[i]
                var_dtype_list.append((arg, dtype))
        return var_dtype_list

    def cosine_similarity(self, t1, t2):
        t1_norm = (F.normalize(t1, dim=0) * 99).to(torch.int)
        t2_norm = (F.normalize(t2, dim=0) * 99).to(torch.int)

        t1_tenosor = torch.zeros(100)
        t2_tenosor = torch.zeros(100)
        try:
            t1_tenosor[t1_norm] = 0.1
        except IndexError:
            raise IndexError()
        t2_tenosor[t2_norm] = 0.1
        # Compute dot product
        dot_product = torch.dot(t1_tenosor, t2_tenosor)
        total = min(torch.dot(t2_tenosor, t2_tenosor),
                    torch.dot(t1_tenosor, t1_tenosor))
        res = dot_product / total
        return res

    def inv_new_const(self, const_type, const_value):
        if len(const_value) == 0:
            return None
        const_name = f"{const_type.name}inv{self.invented_consts_number}"
        new_const = Const(const_name, const_type, const_value)
        const_exists = False
        for const in self.consts:
            if const.values is not None:
                similarity = self.cosine_similarity(const.values, new_const.values)
                if similarity > 0.9:
                    # integrate range
                    const.values = torch.cat((const.values, new_const.values),
                                             dim=0).unique()
                    const_exists = True
                    break
        if not const_exists:
            self.consts.append(new_const)
            self.invented_consts_number += 1
            return new_const
        else:
            return const

    def remove_primitive_consts(self):
        consts = []
        for const in self.consts:
            if 'phi' in const.name or 'rho' in const.name:
                if const.values is not None:
                    consts.append(const)
            else:
                consts.append(const)
        self.consts = consts

    def get_by_dtype(self, dtype):
        """Get constants that match given dtypes.

        Args:
            dtype (DataType): The data type.

        Returns:
            List of constants whose data type is the given data type.
        """
        consts = []
        for c in self.consts:
            if c.dtype == dtype:
                consts.append(c)
        return consts

    def get_by_dtype_name(self, dtype_name):
        """Get constants that match given dtype name.

        Args:
            dtype_name (str): The name of the data type to be used.

        Returns:
            List of constants whose datatype has the given name.
        """
        consts = []
        for c in self.consts:
            if c.dtype.name == dtype_name:
                consts.append(c)
        return consts

    def term_index(self, term):
        """Get the index of a term in the language.

        Args:
            term (Term): The term to be used.

        Returns:
            int: The index of the term.
        """
        terms = self.get_by_dtype(term.dtype)
        return terms.index(term)

    def get_const_by_name(self, const_name):
        """Get the constant by its name.

        Args:
            const_name (str): The name of the constant.

        Returns:
            Const: The matched constant with the given name.

        """
        const = [c for c in self.consts if const_name == c.name]
        assert len(const) == 1, 'Too many match in ' + const_name
        return const[0]

    def get_pred_by_name(self, pred_name):
        """Get the predicate by its name.

        Args:
            pred_name (str): The name of the predicate.

        Returns:
            Predicate: The matched preicate with the given name.
        """
        pred = [pred for pred in self.preds if pred.name == pred_name]
        if not len(pred) == 1:
            print("")
        return pred[0]

    def get_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.invented_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) == 1:
            raise ValueError('Too many or less match in ' + invented_pred_name)
        return invented_pred[0]

    def get_bk_invented_pred_by_name(self, invented_pred_name):
        """Get the predicate by its name.

        Args:
            invented_pred_name (str): The name of the predicate.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        invented_pred = [invented_pred for invented_pred in self.bk_inv_preds if
                         invented_pred.name == invented_pred_name]
        if not len(invented_pred) > 0:
            raise ValueError('Too less match in ' + invented_pred_name)
        return invented_pred[0]

    # def inv_pred(self, args, arity, pi_dtypes, p_args, pi_type):
    #     """Get the predicate by its id.
    #
    #     Args:
    #         pi_template (str): The name of the predicate template.
    #
    #     Returns:
    #         InventedPredicat: The matched invented predicate with the given name.
    #     """
    #     prefix = "inv_pred"
    #     new_predicate_id = self.invented_preds_number
    #     if args is not None:
    #         args.p_inv_counter += 1
    #         self.invented_preds_number = args.p_inv_counter
    #     pred_with_id = prefix + str(new_predicate_id)
    #
    #     new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_type=pi_type)
    #     # self.invented_preds.append(new_predicate)
    #
    #     return new_predicate

    def load_inv_pred(self, id, arity, pi_dtypes, p_args, pi_type):
        """Get the predicate by its id.

        Args:
            pi_template (str): The name of the predicate template.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        prefix = "inv_pred"
        # new_predicate_id = self.invented_preds_number

        pred_with_id = prefix + str(id)

        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes,
                                          p_args, pi_type=pi_type)
        # self.invented_preds.append(new_predicate)

        return new_predicate

    def append_new_predicate(self, old_predicates, new_predicates):
        for new_predicate in new_predicates:
            if new_predicate not in old_predicates:
                old_predicates.append(new_predicate)
        return old_predicates

    # def load_minimum(self, neural_pred=None, full_bk=True):
    #
    #     if neural_pred is not None:
    #         self.preds = self.append_new_predicate(self.preds, neural_pred)
    #     self.invented_preds = list(set(self.all_invented_preds))
    #     self.preds = self.append_new_predicate(self.preds, self.invented_preds)
    #     self.pi_clauses = list(set(self.all_pi_clauses))
    #
    #     prim_args_list = []
    #     for c in self.all_clauses:
    #         for atom in c.body:
    #             if atom.pred.pi_type == "clu_pred":
    #                 for pi_body in atom.pred.body:
    #                     for atom in pi_body:
    #                         prim_args_list.append(atom.terms[:-1])
    #             else:
    #                 prim_args_list.append(atom.terms[:-1])
    #     prim_args_list = list(set(prim_args_list))
    #     self.generate_minimum_atoms(prim_args_list)

    def update_bk(self):
        self.generate_atoms()

    def variable_set_id(self, args, var_id):
        for c in self.clauses:
            for a_i in range(len(c.body)):
                if isinstance(c.body[a_i], InvAtom):
                    c.body[a_i].terms = list(c.body[a_i].terms)
                    for t_i in range(len(c.body[a_i].terms)):
                        if isinstance(c.body[a_i].terms[t_i], Var):
                            c.body[a_i].terms = list(c.body[a_i].terms)
                            if bk.variable_symbol_group in c.body[a_i].terms[
                                t_i].name:
                                c.body[a_i].terms[t_i] = Var(
                                    f"{bk.variable_symbol_group}_{var_id}",
                                    bk.var_dtypes["group"])
                            c.body[a_i].terms = tuple(c.body[a_i].terms)
                        c.body[a_i].terms = tuple(c.body[a_i].terms)

                else:
                    for t_i in range(len(c.body[a_i].terms)):
                        c.body[a_i].terms = list(c.body[a_i].terms)
                        if isinstance(c.body[a_i].terms[t_i], Var):
                            if bk.variable_symbol_group in c.body[a_i].terms[
                                t_i].name:
                                c.body[a_i].terms[t_i] = Var(
                                    f"{bk.variable_symbol_group}_{var_id}",
                                    bk.var_dtypes["group"])
                        c.body[a_i].terms = tuple(c.body[a_i].terms)
        args.logger.debug(
            f"\nAll {len(self.clauses)} Machine Clauses in Group {var_id}:" +
            f"".join([f"\n{str(c)}" for c in self.clauses]))

    def record_milestone(self):
        self.record_clauses += self.clauses
        self.record_atoms += self.atoms
        self.record_consts += self.occurred_consts
        self.record_predicates += self.predicates
        self.record_group_variable_num += 1

    def clear_repeat_language(self):
        self.clauses = list(set(self.record_clauses))
        self.atoms = list(set(self.record_atoms))
        self.consts = list(set(self.record_consts))
        self.predicates = list(set(self.record_predicates))
        self.group_variable_num = self.record_group_variable_num

        _, self.min_consts = self.load_consts(self.number, self.group_variable_num, self.phi_num,
                                              self.rho_num, 1)
        for min_const in self.min_consts:
            if min_const not in self.consts:
                self.consts.append(min_const)
        self.group_vars = [
            Var(f"{self.variable_group_symbol}_{v_i}", bk.var_dtypes["group"]) for v_i in
            range(self.group_variable_num)]
        self.generate_atoms()

    def rewrite_clauses(self, args):
        rewritted_clauses = []
        for g_i in range(self.group_variable_num):
            predicate_list = []
            obj_term_lists = []
            group_term_lists = []
            pattern_term_lists = []
            terms = []
            for clause in self.clauses:
                if clause.body[0].terms[-2].name.split("_")[-1] != str(g_i):
                    continue
                for atom in clause.body:
                    if isinstance(atom, InvAtom):
                        # new term
                        predicate_list.append(atom.pred)
                        obj_term_lists.append(
                            lang_utils.filter_given_type_of_terms(atom.terms,
                                                                  "object"))
                        group_term_lists += lang_utils.filter_given_type_of_terms(
                            atom.terms, "group")
                # invent predicate for rephased clauses
            group_term_lists = list(set(group_term_lists))
            inv_p = FinalPredicate(predicate_list, "exist", 3)
            terms = [obj_term_lists, group_term_lists, clause.head.terms]
            inv_atom = InvAtom(inv_p, terms)
            body = [inv_atom]
            rewritted_clauses.append(Clause(self.clauses[0].head, body))

        args.logger.debug(
            f" \n =============== Learned Machine Clauses =================" + "".join(
                [f"\n{c_i + 1}/{len(self.clauses)} {self.clauses[c_i]}" for c_i in
                 range(len(self.clauses))]))

        args.logger.debug(
            f" \n =============== Rewrote and Merged Clauses =================" + "".join(
                [f"\n{c_i + 1}/{len(rewritted_clauses)} {rewritted_clauses[c_i]}" for
                 c_i in
                 range(len(rewritted_clauses))]))

        return rewritted_clauses

    def update_predicates(self, clauses):
        for clause in clauses:
            if clause.body[0].pred not in self.predicates:
                self.predicates.append(clause.body[0].pred)

