# Created by X at 25.06.24
from src import bk


def get_pi_bodies_by_name(pi_clauses, pi_name):
    pi_bodies_all = []
    for pi_c in pi_clauses:
        if pi_name == pi_c.head.pred.name:
            pi_bodies = []
            for b in pi_c.body:
                p_name = b.pred.name
                if "inv_pred" in p_name:
                    body_names = get_pi_bodies_by_name(pi_clauses, p_name)
                    pi_bodies += body_names
                else:
                    pi_bodies.append(b)
            pi_bodies_all += pi_bodies

    return pi_bodies_all


def check_repeat_conflict(atom1, atom2):
    if atom1.terms[0].name == atom2.terms[0].name and atom1.terms[1].name == atom2.terms[1].name:
        return True
    if atom1.terms[0].name == atom2.terms[1].name and atom1.terms[1].name == atom2.terms[0].name:
        return True
    return False


def is_conflict_bodies(pi_bodies, clause_bodies):
    is_conflict = False
    for i, p_b in enumerate(pi_bodies):
        for j, c_b in enumerate(clause_bodies):
            if p_b == c_b and p_b.pred.name != "in":
                is_conflict = True
            elif p_b.pred.name == c_b.pred.name:
                if p_b.pred.name == "rho":
                    is_conflict = check_repeat_conflict(p_b, c_b)
                elif p_b.pred.name == "phi":
                    is_conflict = check_repeat_conflict(p_b, c_b)
            if is_conflict:
                return True
    return False


def get_c_head(pred, arguments):
    head_str = f"{pred.split(':')[0]}({arguments})"
    return head_str


def merge_term_lists(term_lists):
    for t_i in range(len(term_lists)):
        term_lists[t_i] = tuple(set([t for t_tuple in term_lists[t_i] for t in t_tuple]))
    return term_lists
def remove_chaos_terms(inv_atom_terms):
    # guarantee only one var term in each term list
    rational_terms = []
    for term_list in inv_atom_terms:
        var_terms = []
        for terms in term_list:
            for term in terms:
                if "group_data" in term.name:
                    if term.name not in var_terms:
                        var_terms.append(term.name)
        if len(var_terms) == 1:
            rational_terms.append(term_list)
            # for t in term_list:
            #     if t not in rational_terms:
            #         rational_terms.append(t)
    return rational_terms


def get_var_type(var_str):
    if bk.variable_symbol_group in var_str:
        var_type = bk.var_dtypes["group"]
    elif bk.variable_symbol_obj in var_str:
        var_type = bk.var_dtypes["object"]
    elif bk.variable_symbol_pattern in var_str:
        var_type = bk.var_dtypes["pattern"]
    else:
        raise ValueError
    return var_type


# def inv_new_atom_terms(preds, terms):
#     # invent exist terms
#     objs_terms = []
#
#     for term in terms:
#         if term.dtype.name
#
#     exist_p_idx = [i for i in range(len(preds)) if "exist_obj" in preds[i].name]
#     exist_terms = [terms[i] for i in exist_p_idx]
#     term_objs = [t[0] for t in exist_terms]
#
#
#     exist_group_p_idx = [i for i in range(len(preds)) if "exist_group" in preds[i].name]
#     exist_group_preds = [preds[i] for i in exist_group_p_idx]
#     exist_group_terms = [terms[i] for i in exist_group_p_idx]
#     term_groups = [t[0] for t in exist_group_terms]
#
#
#     return term_objs, term_groups


def orgnize_inv_pred_dtypes(dtypes):
    reorgnized_dtypes = []
    for dt in dtypes:
        reorgnized_dtypes +=dt
        # reorgnized_dtypes.append(dt[0])
    # reorgnized_dtypes.append(dtypes[0][1])
    # reorgnized_dtypes.append(dtypes[0][2])
    # reorgnized_dtypes =
    return reorgnized_dtypes


def filter_given_type_of_terms(terms, term_type):
    type_terms = []
    for term in terms:
        if hasattr(term, "dtype") and term_type in term.dtype.name:
            type_terms.append(term)
        elif hasattr(term, "var_type") and term_type in term.var_type:
            type_terms.append(term)
    return type_terms