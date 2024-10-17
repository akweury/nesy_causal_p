# Created by jing at 25.06.24
from . import bk

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