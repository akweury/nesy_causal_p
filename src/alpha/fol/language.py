# Created by shaji at 24/06/2024
import glob
from lark import Lark
import itertools
import re
import torch.nn.functional as F
import torch

from .exp_parser import ExpTree
from .logic import *
from . import bk, lang_utils, mode_declaration
import config


def get_unused_args(c):
    unused_args = []
    used_args = []
    for body in c.body:
        if "in" == body.pred.name:
            unused_args.append(body.terms[0])
    for body in c.body:
        if not "in" in body.pred.name:
            for term in body.terms:
                if bk.variable["group"] in term.name and term not in used_args:
                    unused_args.remove(term)
                    used_args.append(term)
    return unused_args, used_args


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

    def __init__(self, args, inv_consts=None):

        # BK
        self.vars = [Var(f"{v_name}_{v_i}") for v_i in range(args.g_num) for v_name in bk.variable["group"]]
        self.var_num = args.g_num
        self.atoms = []
        self.funcs = []
        self.consts = []
        self.preds = []
        self.mode_declarations = None
        # PI
        self.inv_p = []
        self.inv_p_with_scores = []
        self.all_inv_p = []
        self.pi_c = []
        self.all_pi_c = []
        self.clause_with_scores = []
        # self.invented_preds_number = args.p_inv_counter
        self.invented_consts_number = 0

        # Results
        self.learned_c = []

        # GRAMMAR
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")
        with open(args.lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")

        # load BK predicates and constants
        level_0_preds = [
            self.parse_pred(bk.predicate_target),
            self.parse_pred(bk.predicate_exist)
        ]
        level_1_preds = [self.parse_pred(data) for data in list(bk.neural_p.values())]
        level_2_preds = [self.parse_inv_pred(inv_p_i) for inv_p_i in range(args.g_num)]
        self.preds = level_0_preds + level_1_preds + level_2_preds
        self.consts = self.load_consts(args)

    def reset_lang(self, group_num):
        self.learned_c = []
        init_c = self.load_init_clauses(group_num)
        # update predicates
        self.update_bk()
        # update language
        self.mode_declarations = mode_declaration.get_mode_declarations(self.preds, group_num)
        return init_c

    def __str__(self):
        s = "===Predicates===\n"
        for pred in self.preds:
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

    def generate_minimum_atoms(self, prim_args_list):
        p_ = Predicate('.', 1, [mode_declaration.DataType('spec')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype, with_inv=True) for dtype in dtypes]
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
            consts_list = [self.get_by_dtype(dtype, with_inv=True) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                # check if args and pred correspond are in the same area
                if pred.dtypes[0].name == 'area':
                    if pred.name[0] + pred.name[5:] != args[0].name:
                        continue
                if len(args) == 1 or len(set(args)) == len(args):
                    pi_atoms.append(Atom(pred, args))
        self.atoms = spec_atoms + sorted(atoms) + sorted(bk_pi_atoms) + sorted(pi_atoms)

    def generate_atoms(self):
        p_ = Predicate('.', 1, [DataType('spec')])
        false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
        true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

        spec_atoms = [false, true]
        atoms = []
        for pred in self.preds:
            dtypes = pred.dtypes
            consts_list = [self.get_by_dtype(dtype) for dtype in dtypes]
            # if pred.pi_type == "clu_pred":
            #     consts_list = [[atom.terms[0]] for atom in pred.body[0]]
            args_list = list(set(itertools.product(*consts_list)))
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    atoms.append(Atom(pred, args))
        # pi_atoms = []
        # for pred in self.invented_preds:
        #     consts_list = []
        #     for body_pred in pred.body[0]:
        #         consts_list.append([body_pred.terms[0]])
        #
        #     args_list = list(set(itertools.product(*consts_list)))
        #     for args in args_list:
        #         new_atom = Atom(pred, args)
        #         if new_atom not in atoms:
        #             pi_atoms.append(new_atom)

        # bk_pi_atoms = []
        # for pred in self.bk_inv_preds:
        #     dtypes = pred.dtypes
        #     consts_list = [self.get_by_dtype(dtype, with_inv=True) for dtype in dtypes]
        #     args_list = list(set(itertools.product(*consts_list)))
        #     for args in args_list:
        #         # check if args and pred correspond are in the same area
        #         if pred.dtypes[0].name == 'area':
        #             if pred.name[0] + pred.name[5:] != args[0].name:
        #                 continue
        #         if len(args) == 1 or len(set(args)) == len(args):
        #             pi_atoms.append(Atom(pred, args))
        self.atoms = spec_atoms + sorted(atoms)  # + sorted(bk_pi_atoms) + sorted(pi_atoms)

    def load_init_clauses(self, g_num):
        """Read lines and parse to Atom objects.
        """

        # final target clause
        head = [f"inv_p{i}(X)" for i in range(g_num)]
        group_clauses_str = [head[h_i] + f":-in({bk.variable['group'][h_i]}_0,X)." for h_i in range(len(head))]

        # target(X):-in(G1,X),in(G2,X).
        # in(G1,X):-color_map(ig1, og1, rr).
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
        dtype_names = dtype_names_str.split(',')
        dtypes = [mode_declaration.DataType(dt) for dt in dtype_names]
        return NeuralPredicate(head_str, int(arity), dtypes)

    def parse_inv_pred(self, inv_p_id):
        head = f"inv_p{inv_p_id}"
        arity = 1
        head_dtype_names = ['pattern']
        dtypes = [mode_declaration.DataType(dt) for dt in head_dtype_names]

        # pred_with_id = pred + f"_{i}"
        pred_with_id = head.split("(")[0]
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)
        return invented_pred

    def parse_const(self, args, const, const_type):
        """Parse string to function symbols.
        """
        const_data_type = mode_declaration.DataType(const)
        if "amount_" in const_type:
            _, num = const_type.split('_')
            if num == 'e':
                num = args.g_num
            elif num == "phi":
                num = args.phi_num
            elif num == "rho":
                num = args.rho_num
            elif num == "slope":
                num = args.slope_num
            elif num == "num":
                num = args.number_num
            const_names = []
            for i in range(int(num)):
                # if const == "group" and i == 0:
                #     continue
                const_names.append(f"{const}{i + 1}of{num}")
        elif "enum" in const_type:
            if const == 'color':
                const_names = bk.color
            elif const == 'shape':
                const_names = bk.shape
            else:
                raise ValueError
        elif 'target' in const_type:
            const_names = ['pattern']
        else:
            raise ValueError

        return [Const(const_name, const_data_type) for const_name in const_names]

    def load_consts(self, args):
        consts_str = []
        for const_name, const_type in bk.const_dict.items():
            consts_str.extend(self.parse_const(args, const_name, const_type))
        return consts_str

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
        invented_pred = InventedPredicate(pred_with_id, int(arity), dtypes, args=None, pi_type=None)

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
        total = min(torch.dot(t2_tenosor, t2_tenosor), torch.dot(t1_tenosor, t1_tenosor))
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
                    const.values = torch.cat((const.values, new_const.values), dim=0).unique()
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
                if c.values is None:
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
                if c.values is None:
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

    def inv_pred(self, args, arity, pi_dtypes, p_args, pi_type):
        """Get the predicate by its id.

        Args:
            pi_template (str): The name of the predicate template.

        Returns:
            InventedPredicat: The matched invented predicate with the given name.
        """
        prefix = "inv_pred"
        new_predicate_id = self.invented_preds_number
        if args is not None:
            args.p_inv_counter += 1
            self.invented_preds_number = args.p_inv_counter
        pred_with_id = prefix + str(new_predicate_id)

        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_type=pi_type)
        # self.invented_preds.append(new_predicate)

        return new_predicate

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

        new_predicate = InventedPredicate(pred_with_id, int(arity), pi_dtypes, p_args, pi_type=pi_type)
        # self.invented_preds.append(new_predicate)

        return new_predicate

    def append_new_predicate(self, old_predicates, new_predicates):
        for new_predicate in new_predicates:
            if new_predicate not in old_predicates:
                old_predicates.append(new_predicate)
        return old_predicates

    def load_minimum(self, neural_pred=None, full_bk=True):

        if neural_pred is not None:
            self.preds = self.append_new_predicate(self.preds, neural_pred)
        self.invented_preds = list(set(self.all_invented_preds))
        self.preds = self.append_new_predicate(self.preds, self.invented_preds)
        self.pi_clauses = list(set(self.all_pi_clauses))

        prim_args_list = []
        for c in self.all_clauses:
            for atom in c.body:
                if atom.pred.pi_type == "clu_pred":
                    for pi_body in atom.pred.body:
                        for atom in pi_body:
                            prim_args_list.append(atom.terms[:-1])
                else:
                    prim_args_list.append(atom.terms[:-1])
        prim_args_list = list(set(prim_args_list))
        self.generate_minimum_atoms(prim_args_list)

    def update_bk(self):
        # put everything into the bk
        # self.preds = self.append_new_predicate(self.preds, self.preds)
        # self.invented_preds = self.all_invented_preds
        # self.preds = self.append_new_predicate(self.preds, self.invented_preds)
        # self.pi_clauses = self.all_pi_clauses
        self.generate_atoms()
