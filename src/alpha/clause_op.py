# Created by jing at 11.10.24

from src.alpha.fol.logic import Var, Clause, InventedPredicate, Atom, InvAtom



def non_trivial_vars(merged_body):
    vars = []
    for atom in merged_body:
        if atom.pred.name not in ["ing", "inp", "target"]:
            vars += atom.all_vars()
    vars = list(set(vars))
    return vars


def inv_pred_merge_bodies(bodies):
    # new predicate
    # pred_names = [b.pred.name for b in bodies]
    # args_list = [b.pred.dtypes for b in bodies]
    # sorted_indices = sorted(range(len(pred_names)), key=lambda i: pred_names[i])
    # pred_names = [pred_names[i] for i in sorted_indices]
    # args_list = [args_list[i] for i in sorted_indices]
    preds = []
    for atom in bodies:
        if isinstance(atom, Atom):
            preds.append(atom.pred)
        elif isinstance(atom, InvAtom):
            preds+=atom.pred.sub_preds
    preds = list(set(preds))

    pred_indices = sorted(range(len(preds)), key=lambda i: preds[i])
    preds = [preds[i] for i in pred_indices]
    dtypes = [dt for pred in preds for dt in pred.dtypes]
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
        trivial_body = [b for b in bodies if b.pred.name in ["ing", "inp", "target"]]
        new_body = [b for b in bodies if b.pred.name not in ["ing", "inp", "target"]]
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
