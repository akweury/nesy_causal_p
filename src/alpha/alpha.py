# Created by X at 24/06/2024
import itertools
import torch
import copy
from src.utils import log_utils
from src.alpha import valuation, facts_converter, nsfr, pruning, clause_op
from src.alpha.fol import language, refinement
from src import bk


def init_ilp(args, obj_num):
    args.variable_group_symbol = bk.variable_symbol_group
    args.variable_obj_symbol = bk.variable_symbol_obj
    lang = language.Language(obj_num, args.variable_group_symbol,
                             args.variable_obj_symbol, args.lark_path,
                             args.phi_num, args.rho_num, args.obj_n)
    return lang


def extension(args, lang, clauses):
    refs = []
    B_ = []

    refinement_generator = refinement.RefinementGenerator(lang=lang)
    for c_original in clauses:
        c = copy.deepcopy(c_original)
        refs_i = refinement_generator.refinement_clause(c)
        unused_args, used_args = language.get_unused_args(c)
        # refs_i_removed = pruning.remove_duplicate_clauses(refs_i, unused_args, used_args, args)
        # remove already appeared refs
        refs_i_removed = list(set(refs_i).difference(set(B_)))
        B_.extend(refs_i_removed)
        refs.extend(refs_i_removed)

    # remove semantic conflict clauses
    # refs_no_conflict = pruning.remove_conflict_clauses(refs, lang.pi_clauses, args)
    # if len(refs) == 0:
    #     refs_no_conflict = clauses
    #     args.is_done = True

    if args.show_process:
        log_utils.add_lines(f"=============== extended clauses =================",
                            args.log_file)
        for ref in refs:
            log_utils.add_lines(f"{ref}", args.log_file)
    if len(refs) == 0:
        print("No extended clauses found")
    return refs


def node_extension(args, lang, base_nodes, extended_nodes):
    new_nodes = []
    node_combs = list(set(itertools.product(base_nodes, extended_nodes)))
    for node_comb in node_combs:
        clause_combined, new_atoms = clause_op.merge_clauses(node_comb, lang)
        if len(clause_combined) > 0:
            for c in clause_combined:
                if c not in new_nodes:
                    new_nodes.append(c)
            for a_i in range(len(new_atoms)):
                if new_atoms[a_i] not in lang.atoms:
                    lang.atoms.append(new_atoms[a_i])

    return new_nodes


def eval_ims(NSFR, args, pred_names, objs):
    """ input: clause, output: score """

    atom_values = NSFR.clause_eval_quick(objs)
    # each score needs an explanation
    score_positive = NSFR.get_target_prediciton(atom_values, pred_names, args.device)
    # if score_positive.size(2) > 1:
    #     score_positive = score_positive.max(dim=2, keepdim=True)[0]

    return score_positive


def eval_dls(ils):
    # dls = ils.sum(dim=1) / ils.shape[1]
    dls = ils.max(dim=1)[0]
    return dls.squeeze()


def sort_clauses_by_score(clauses, scores_all, scores):
    clause_with_scores = []
    for c_i, clause in enumerate(clauses):
        score = scores[:, c_i]
        clause_with_scores.append((clause, score, scores_all[c_i]))

    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][2], reverse=True)
        return c_sorted

    return clause_with_scores


def evaluation(args, NSFR, target_preds, objs):
    # image level scores
    ils = eval_ims(NSFR, args, target_preds, objs)
    # dataset level scores
    dls = eval_dls(ils)
    return ils, dls


def beam_search(args, lang, C, FC, objs):
    clauses = C
    for bs_step in range(args.max_bs_step):
        # clause extension
        extended_C = extension(args, lang, clauses)
        if args.is_done:
            break
        NSFR = nsfr.get_nsfr_model(args, lang, FC, extended_C)
        target_preds = list(set([c.head.pred.name for c in extended_C]))
        # clause evaluation
        ils, dls = evaluation(args, NSFR, target_preds, objs)
        # prune clauses
        pruned_c = pruning.top_k_clauses(args, ils, dls, extended_C)
        # save data
        if len(pruned_c) == 0:
            break
        else:
            clauses = pruned_c
        print(f"Target Clauses Num : {len(clauses)}, Step: {bs_step}")
    return clauses


def df_search(args, lang, C, FC, group, group_num, level):
    # node evaluation
    lang.reset_lang(g_num=group_num, level=level)
    atom_C = extension(args, lang, C)
    NSFR = nsfr.get_nsfr_model(args, lang, FC, atom_C)
    target_preds = list(set([c.head.pred.name for c in atom_C]))
    # clause evaluation
    ils, dls = evaluation(args, NSFR, target_preds, group)
    # node extension (DFS)
    base_nodes = [atom_C[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
    extended_nodes = [atom_C[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
    # update const lists
    lang.update_consts(base_nodes)
    lang.generate_atoms()

    if len(base_nodes) == 0:
        return
    elif len(extended_nodes) == 1:
        lang.clauses += extended_nodes
        return

    extended_nodes = node_extension(args, lang, base_nodes, extended_nodes)
    NSFR = nsfr.get_nsfr_model(args, lang, FC, extended_nodes)
    target_preds = list(set([c.head.pred.name for c in extended_nodes]))
    # clause evaluation
    ils, dls = evaluation(args, NSFR, target_preds, group)
    pass_indices = [s_i for s_i in range(len(ils)) if ils[s_i] > 0.8]
    extended_nodes = [extended_nodes[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
    ils = ils[pass_indices]

    log_clause_str = ""
    for i in range(len(ils)):
        log_clause_str += f"\n {i + 1}/{len(ils)} (s: {ils[i].item():.2f}) Clause: {extended_nodes[i]}"
    args.logger.debug(log_clause_str)

    extended_nodes = sorted(extended_nodes) + sorted(base_nodes)
    lang.clauses += extended_nodes
    clauses = lang.clauses
    return clauses


def align_ocms(ocms):
    aligned_ocms = []
    max_len = max([len(ocm) for ocm in ocms])
    for ocm in ocms:
        if len(ocm) < max_len:
            ocm = torch.cat((ocm, torch.zeros(max_len - len(ocm), 10)))
        aligned_ocms.append(ocm)
    return aligned_ocms


def eval_task(args, lang, FC, images_data, all_clauses, level):
    """
    return: true, if all the clauses are satisfied, else false.
    """
    lang.clauses = all_clauses
    target_preds = list(set([c.head.pred.name for c in lang.clauses]))
    preds = []
    for example_i in range(len(images_data)):
        example_preds = []
        for clause in all_clauses:
            NSFR = nsfr.get_nsfr_model(args, lang, FC, [clause])
            clause_preds = []
            for g_i, group in enumerate(images_data[example_i]):
                group_preds = []
                if level == "group":
                    enum_example_groups = []
                    for g_i, g in enumerate(images_data[example_i]):
                        enum_example_groups.append([g_i, g])
                    group_combs = list(itertools.combinations(enum_example_groups, 2))
                    for groups in group_combs:
                        gcms = torch.cat([group[1]["gcm"] for group in groups], dim=0)
                        ocms = [group[1]["ocm"] for group in groups]
                        ocms_aligned = align_ocms(ocms)
                        group_indices = [group[0] for group in groups]
                        data = {"gcms": gcms, "ocms": ocms_aligned}
                        ils, _ = evaluation(args, NSFR, target_preds, data)
                        group_pred = [ils.squeeze()]
                        group_preds.append(group_pred)
                else:
                    ocms = group["ocm"]

                    for o_i, ocm in enumerate(ocms):
                        ils, _ = evaluation(args, NSFR, target_preds, ocm.unsqueeze(0))
                        group_preds.append(ils.squeeze())
                clause_preds.append(group_preds)
            example_preds.append(clause_preds)
        preds.append(example_preds)
    return preds


def remove_trivial_atoms(args, lang, FC, clauses, objs, data):
    lang.trivial_atom_terms = []
    # clause extension
    clauses = extension(args, lang, clauses)
    NSFR = nsfr.get_nsfr_model(args, lang, FC, clauses)
    target_preds = list(set([c.head.pred.name for c in clauses]))
    # clause evaluation
    img_scores, clause_scores = evaluation(args, NSFR, target_preds, objs, data)
    trivial_c = [clauses[i] for i in range(len(clause_scores)) if
                 clause_scores[i] < args.th_inv_nc]
    trivial_atom_terms = []
    for c in trivial_c:
        trivial_atom_terms.append(c.body[0].terms[:-1])
    lang.trivial_atom_terms = trivial_atom_terms

    non_trivial_atoms = []
    for atom in lang.atoms:
        if len(atom.terms) <= 1:
            non_trivial_atoms.append(atom)
        elif atom.terms[:-1] not in trivial_atom_terms:
            non_trivial_atoms.append(atom)

    return non_trivial_atoms


# def search_clauses(args, ocm, gcm, groups):
#     lang = None
#     example_num = len(groups)
#     principle_num = groups.shape[1]
#     all_clauses = []
#     for a_i in range(principle_num):
#         principle_clauses = []
#         principle_gcm = [gcm[i][a_i] for i in range(len(gcm))]
#         same_length = all([len(_gcm) == len(principle_gcm[0]) for _gcm in principle_gcm])
#         if not same_length:
#             continue
#
#         for e_i in range(example_num):
#             # reasoning clauses
#             lang = alpha(args, ocm[e_i], principle_gcm[e_i], groups[e_i, a_i, :len(ocm[e_i])])
#             if len(lang.clauses) == 0:
#                 break
#             principle_clauses += lang.clauses
#         # remove infrequent clauses
#         if len(principle_clauses) > 0:
#             principle_clauses, lang = filter_infrequent_clauses(principle_clauses, lang, example_num)
#         all_clauses += principle_clauses
#     return lang


def common_elements(lists):
    if not lists:
        return set()

    # Convert the first list to a set and intersect with the rest
    common = set(lists[0])
    for lst in lists[1:]:
        common &= set(lst)

    return common


# def check_clause_in_images(clause, all_clauses):
#     exist_all = True
#     for image_clauses in all_clauses:
#         for group in image_clauses:
#             for group_clause in group["group_clauses"]:
#                 if clause
def search_common_clauses(all_groups):
    common_group_clauses = []
    # test the clause in the first image, if it exists in the rest of images, save it
    for group in all_groups[0]:
        for group_clause in group["group_clauses"]:
            exist = check_clause_in_images(group_clause, all_groups[1:])
            if exist:
                common_group_clauses.append(group_clause)


def save_lang(args, lang, mode, level):
    lang_dict = {
        "all_groups": lang.all_groups,
        "atoms": lang.atoms,
        "clauses": lang.clauses,
        "consts": lang.consts,
        "preds": lang.predicates,
        "g_num": lang.group_variable_num,
        "attrs": lang.attrs,
    }
    torch.save(lang_dict, str(args.output_file_prefix) + f'learned_lang_{mode}_{level}.pkl')


def alpha(args, groups, mode):
    obj_num = 1
    lang = init_ilp(args, obj_num)
    lang.reset_lang(g_num=1)
    VM = valuation.get_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM)
    C = lang.load_obj_init_clauses()

    example_num = len(groups)
    all_groups = []
    for example_i in range(example_num):
        example_groups = []
        for g_i, group in enumerate(groups[example_i]):
            ocms = group["ocm"]
            gcm = group["gcm"]
            obj_clauses = {}
            for o_i, ocm in enumerate(ocms):
                clauses = df_search(args, lang, C, FC, ocm.unsqueeze(0))
                if clauses is None:
                    continue
                for clause in clauses:
                    clause = clause_op.change_clause_obj_id(clause, args, o_i, bk.variable_symbol_obj)
                    if clause not in obj_clauses:
                        obj_clauses[clause] = 1
                    else:
                        obj_clauses[clause] += 1
            group_clauses = df_search(args, lang, C, FC, gcm)
            gcs = {}
            for clause in group_clauses:
                clause = clause_op.change_clause_obj_id(clause, args, g_i, bk.variable_symbol_group)
                if clause not in gcs:
                    gcs[clause] = 1
                else:
                    gcs[clause] += 1
            group_data = {"group_clauses": gcs, "obj_clauses": obj_clauses}
            example_groups.append(group_data)
        all_groups.append(example_groups)
    lang.all_groups = all_groups
    # update language consts, atoms
    g_clauses = []
    o_clauses = []
    for ic in all_groups:
        for g in ic:
            for g_clause, _ in g["group_clauses"].items():
                if g_clause not in g_clauses:
                    g_clauses.append(g_clause)
            for obj_clause, _ in g["obj_clauses"].items():
                if obj_clause not in o_clauses:
                    o_clauses.append(obj_clause)
    lang.update_consts(g_clauses + o_clauses)
    lang.generate_atoms(g_clauses + o_clauses)
    lang.update_predicates(g_clauses + o_clauses)
    lang.clauses = g_clauses + o_clauses

    save_lang(args, lang, mode)
    return lang


def alpha_object(args, all_matrix, mode):
    obj_num = 1
    lang = init_ilp(args, obj_num)
    lang.reset_lang(g_num=1, level="object")
    VM = valuation.get_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM)
    C = lang.load_obj_init_clauses()
    example_num = len(all_matrix)
    all_groups = []
    for example_i in range(example_num):
        example_groups = []
        for g_i, group in enumerate(all_matrix[example_i]):
            ocms = group["ocm"]
            # gcm = group["gcm"]
            obj_clauses = {}
            for o_i, ocm in enumerate(ocms):
                clauses = df_search(args, lang, C, FC, ocm.unsqueeze(0), 1, level="object")
                if clauses is None:
                    continue
                for clause in clauses:
                    clause = clause_op.change_clause_obj_id(clause, args, o_i, bk.variable_symbol_obj)
                    if clause not in obj_clauses:
                        obj_clauses[clause] = 1
                    else:
                        obj_clauses[clause] += 1
            group_data = {"obj_clauses": obj_clauses}
            example_groups.append(group_data)
        all_groups.append(example_groups)
    lang.all_groups = all_groups
    # update language consts, atoms
    o_clauses = []
    for ic in all_groups:
        for g in ic:
            for obj_clause, _ in g["obj_clauses"].items():
                if obj_clause not in o_clauses:
                    o_clauses.append(obj_clause)
    lang.update_consts(o_clauses)
    lang.generate_atoms(o_clauses)
    lang.update_predicates(o_clauses)
    lang.clauses = o_clauses
    save_lang(args, lang, mode, "object")
    return lang


def alpha_group(args, all_matrix, mode):
    example_num = len(all_matrix)
    all_clauses = []
    for example_i in range(example_num):
        group_num = len(all_matrix[0])
        lang = init_ilp(args, group_num)
        lang.reset_lang(g_num=group_num, level="group")
        VM = valuation.get_group_valuation_module(args, lang)
        FC = facts_converter.FactsConverter(args, lang, VM)
        C = lang.load_group_init_clauses()
        example_clauses = []
        example_groups = all_matrix[example_i]
        enum_example_groups = []
        for g_i, g in enumerate(example_groups):
            enum_example_groups.append([g_i, g])
        group_combs = list(itertools.combinations(enum_example_groups, 1))
        for groups in group_combs:
            gcms = torch.cat([group[1]["gcm"] for group in groups], dim=0)
            max_o_num = max([len(g[1]["ocm"]) for g in groups])
            ocms = []
            for group in groups:
                if len(group[1]["ocm"]) < max_o_num:
                    ocms.append(
                        torch.cat((group[1]["ocm"], torch.zeros((max_o_num - len(group[1]["ocm"]), 10))), dim=0))
                else:
                    ocms.append(group[1]["ocm"])
            ocms = torch.stack(ocms)
            group_indices = [group[0] for group in groups]
            data = {"gcms": gcms, "ocms": ocms}
            group_clauses = df_search(args, lang, C, FC, data, group_num, level="group")
            gcs = {}
            for clause in group_clauses:
                # clause = clause_op.change_clause_obj_id(clause, args, group_indices, bk.variable_symbol_group)
                if clause not in gcs:
                    gcs[clause] = 1
                else:
                    gcs[clause] += 1
            group_data = {"group_clauses": gcs}
            example_clauses.append(group_data)
        all_clauses.append(example_clauses)
    removed_all_clauses = remove_trivial_clauses(all_clauses)
    lang.all_groups = removed_all_clauses
    # update language consts, atoms
    g_clauses = []

    for ic in removed_all_clauses:
        for g in ic:
            for g_clause, _ in g["group_clauses"].items():
                if g_clause not in g_clauses:
                    g_clauses.append(g_clause)
    lang.update_consts(g_clauses)
    lang.generate_atoms(g_clauses)
    lang.update_predicates(g_clauses)
    lang.clauses = g_clauses
    save_lang(args, lang, mode, "group")

    return lang


def alpha_test(args, groups, lang, all_clauses, level):
    if level == "object":
        VM = valuation.get_valuation_module(args, lang)
    else:
        VM = valuation.get_group_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM, given_attrs=lang.attrs)
    pred = eval_task(args, lang, FC, groups, all_clauses, level)
    return pred


def is_trivial_clause(c):
    is_trivial = False

    for atom in c.body:
        name = []
        for t in atom.terms:
            if t.name not in name:
                name.append(t.name)
            else:
                is_trivial = True
    return is_trivial


def remove_trivial_clauses(all_clauses):
    non_trivial_clauses = []
    for img_i in range(len(all_clauses)):
        img_clauses = []
        for g_i in range(len(all_clauses[img_i])):
            group_clauses = {"group_clauses": {}}
            for c, count in all_clauses[img_i][g_i]["group_clauses"].items():
                if not is_trivial_clause(c):
                    group_clauses["group_clauses"][c] = count
            img_clauses.append(group_clauses)
        non_trivial_clauses.append(img_clauses)
    return non_trivial_clauses


def filter_infrequent_clauses(all_clauses, lang, example_num):
    frequency = {}
    for item in all_clauses:
        frequency[item] = frequency.get(item, 0) + 1
    most_frequency_value = max(frequency.values())
    most_frequent_clauses = [key for key, value in frequency.items() if
                             value == most_frequency_value]

    if most_frequency_value == example_num:
        lang.done = True
    else:
        lang.done = False
    return most_frequent_clauses, lang
