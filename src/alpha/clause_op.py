# Created by jing at 11.10.24

from src.alpha.fol.logic import Var, Clause, InventedPredicate, Atom


def non_trivial_vars(merged_body):
    vars = []
    for atom in merged_body:
        if atom.pred.name not in ["ing", "inp", "target"]:
            vars += atom.all_vars()
    vars = list(set(vars))
    return vars


def inv_pred_merge_bodies(bodies):
    # new predicate
    pred_names = [b.pred.name for b in bodies]
    args_list = [b.pred.dtypes for b in bodies]
    dtypes = [dt for b in bodies for dt in b.pred.dtypes]
    terms = [t for b in bodies for t in b.terms]

    seen = set()
    new_list1 = []
    new_list2 = []
    for item1, item2 in zip(dtypes, terms):
        if item2 not in seen:
            seen.add(item2)
            new_list1.append(item1)
            new_list2.append(item2)
    dtypes = new_list1

    arity = len(dtypes)
    inv_pred = InventedPredicate('_'.join(pred_names), arity, dtypes, pred_names, args_list)

    return inv_pred


def merge_clauses(clauses, lang):
    # inv_atoms = []
    bodies = [atom for c in clauses for atom in c.body]
    bodies = list(set(bodies))
    vars_in_body = non_trivial_vars(bodies)
    merged_clauses = []
    inv_atoms_grounded = []
    if len(vars_in_body) == 2:
        trivial_body = [b for b in bodies if b.pred.name in ["ing", "inp", "target"]]
        new_body = [b for b in bodies if b.pred.name not in ["ing", "inp", "target"]]
        vars = list(set([t for b in new_body for t in b.terms if type(t) == Var]))
        inv_pred = inv_pred_merge_bodies(new_body)
        inv_atoms_grounded, inv_atoms_ungrounded = lang.generate_inv_atoms(inv_pred, vars)
        head = clauses[0].head
        for inv_atom in inv_atoms_ungrounded:
            body = trivial_body + [inv_atom]
            merged_clauses.append(Clause(head, body))

    return merged_clauses, inv_atoms_grounded
