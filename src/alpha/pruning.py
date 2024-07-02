# Created by jing at 25.06.24
from .fol import lang_utils


def remove_duplicate_clauses(refs_i, unused_args, used_args, args):
    non_duplicate_c = []
    for clause in refs_i:
        is_duplicate = False
        for body in clause.body:
            if "in" != body.pred.name:
                if len(body.terms) == 2 and "O" not in body.terms[1].name:
                    # predicate with 1 object arg
                    if len(unused_args) > 0:
                        if not (body.terms[0] == unused_args[0] or body.terms[0] in used_args):
                            is_duplicate = True
                            break
                # predicate with 2 object args
                elif len(body.terms) == 2 and body.terms[0] in unused_args and body.terms[1] in unused_args:
                    if body.terms[0] not in unused_args[:2] and body.terms[1] not in unused_args:
                        is_duplicate = True
                        break
                elif len(body.terms) == 1 and body.terms[0] in unused_args:
                    if body.terms[0] not in unused_args[:1]:
                        is_duplicate = True
                        break
        if not is_duplicate:
            non_duplicate_c.append(clause)
    return non_duplicate_c


def remove_conflict_clauses(clauses, pi_clauses, args):
    # print("\nCheck for conflict clauses...")
    clause_ordered = []
    non_conflict_clauses = []
    for clause in clauses:
        is_conflict = False
        with_pi = False
        if len(pi_clauses) > 0:
            for cb in clause.body:
                if "inv_pred" in cb.pred.name:
                    with_pi = True
            if not with_pi:
                is_conflict = False
        if with_pi or len(pi_clauses) == 0:
            for i in range(len(clause.body)):
                if is_conflict:
                    break
                for j in range(len(clause.body)):
                    if i == j:
                        continue
                    if "inv_pred" in clause.body[j].pred.name and not is_conflict:
                        pi_name = clause.body[j].pred.name
                        pi_bodies = lang_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = lang_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    if "inv_pred" in clause.body[i].pred.name and not is_conflict:
                        pi_name = clause.body[i].pred.name
                        pi_bodies = lang_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = lang_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
        if not is_conflict:
            non_conflict_clauses.append(clause)

    return non_conflict_clauses


def top_k_clauses(args, ils, dls, clauses):
    ness_score, ness_rank = dls.sort(descending=True)
    top_clauses = [clauses[i] for i in ness_rank[:args.top_k]]
    return top_clauses
