# Created by jing at 11.10.24

from src.alpha.fol.logic import Var, Clause, InventedPredicate, Atom, InvAtom
from src import bk
from src.alpha.fol import lang_utils
def non_trivial_vars(merged_body):
    vars = []
    for atom in merged_body:
        if atom.pred.name not in bk.trivial_preds.keys():
            vars += atom.all_vars()
    vars = list(set(vars))
    return vars


def inv_pred_merge_bodies(bodies):
    preds = []
    for atom in bodies:
        if isinstance(atom, Atom):
            preds.append(atom.pred)
        elif isinstance(atom, InvAtom):
            preds += atom.pred.sub_preds
    preds = list(set(preds))

    if len(preds) == 1:
        return None

    pred_indices = sorted(range(len(preds)), key=lambda i: preds[i])
    preds = [preds[i] for i in pred_indices]
    dtypes = [pred.dtypes for pred in preds]
    dtypes = lang_utils.orgnize_inv_pred_dtypes(dtypes)
    # terms = [t for b in bodies for t in b.terms]
    arity = len(dtypes)
    inv_pred = InventedPredicate(preds, arity, dtypes)

    return inv_pred


def merge_clauses(clauses, lang):
    # inv_atoms = []
    bodies = [atom for c in clauses for atom in c.body]
    bodies = list(set(bodies))
    vars_in_body = non_trivial_vars(bodies)
    merged_clauses = []
    inv_atoms_grounded = []
    inv_atoms_ungrounded = []
    if len(vars_in_body) == 2:
        trivial_body = [b for b in bodies if b.pred.name in list(
            bk.trivial_preds.keys())]
        new_body = [b for b in bodies if b.pred.name not in list(
            bk.trivial_preds.keys())]
        vars = list(set([t for b in new_body for t in b.terms if type(t) == Var]))
        inv_pred = inv_pred_merge_bodies(new_body)
        if inv_pred is not None and inv_pred not in lang.predicates:
            lang.predicates.append(inv_pred)
            inv_atoms_grounded, inv_atoms_ungrounded = lang.generate_inv_atoms(inv_pred, vars)

        head = clauses[0].head
        for inv_atom in inv_atoms_ungrounded:
            body = trivial_body + [inv_atom]
            merged_clauses.append(Clause(head, body))

    return merged_clauses, inv_atoms_grounded
