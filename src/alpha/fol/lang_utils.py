# Created by jing at 25.06.24

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
