# Created by shaji at 24/06/2024
import itertools
import torch

from src.utils import log_utils
from . import valuation, facts_converter, nsfr, pruning, clause_op
from .fol import language
from .fol import refinement
from .fol import bk
from src import llama_call


def init_ilp(args, obj_num):
    args.variable_group_symbol = bk.variable_symbol_group
    args.variable_obj_symbol = bk.variable_symbol_obj
    lang = language.Language(obj_num, args.variable_group_symbol, args.variable_obj_symbol, args.lark_path,
                             args.phi_num, args.rho_num)
    return lang


def load_lang(args, lang_data):
    lang = init_ilp(args, 1)
    lang.clauses = lang_data["clauses"]
    lang.predicates = lang_data["preds"]
    lang.reset_lang(lang_data["g_num"])
    lang.consts = lang_data["consts"]
    lang.atoms = lang_data["atoms"]
    lang.attrs = lang_data["attrs"]
    return lang


def extension(args, lang, clauses):
    refs = []
    B_ = []

    refinement_generator = refinement.RefinementGenerator(lang=lang)
    for c in clauses:
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
        log_utils.add_lines(f"=============== extended clauses =================", args.log_file)
        for ref in refs:
            log_utils.add_lines(f"{ref}", args.log_file)
    if len(refs) == 0:
        print("No extended clauses found")
    return refs


def node_extension(args, lang, base_nodes, extended_nodes):
    # refinement_generator = refinement.RefinementGenerator(lang=lang)
    # extend nodes
    new_nodes = []
    # node_combs = list(itertools.combinations(base_nodes, 2))
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

    if args.show_process:
        log_utils.add_lines(f"=============== extended clauses =================", args.log_file)
        for ref in new_nodes:
            log_utils.add_lines(f"{ref}", args.log_file)
    if len(new_nodes) == 0:
        print("No extended clauses found")
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


def df_search(args, lang, C, FC, objs):
    # node evaluation
    atom_C = extension(args, lang, C)
    NSFR = nsfr.get_nsfr_model(args, lang, FC, atom_C)
    target_preds = list(set([c.head.pred.name for c in atom_C]))
    # clause evaluation
    ils, dls = evaluation(args, NSFR, target_preds, objs)
    # node extension (DFS)
    base_nodes = [atom_C[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
    extended_nodes = [atom_C[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
    # update const lists
    lang.update_consts(base_nodes)

    lang.generate_atoms()
    for e_i in range(2):
        extended_nodes = node_extension(args, lang, base_nodes, extended_nodes)
        try:
            NSFR = nsfr.get_nsfr_model(args, lang, FC, extended_nodes)
        except ValueError:
            raise ValueError
        target_preds = list(set([c.head.pred.name for c in extended_nodes]))
        # clause evaluation
        ils, dls = evaluation(args, NSFR, target_preds, objs)
        pass_indices = [s_i for s_i in range(len(ils)) if ils[s_i] > 0.8]
        extended_nodes = [extended_nodes[s_i] for s_i in range(len(ils)) if ils[s_i] > 0.6]
        ils = ils[pass_indices]
        print(f"inv clauses score: {ils.unique()}")

    # prune clauses
    # pruned_c = pruning.top_k_clauses(args, ils, dls, extended_nodes)
    print(f"extend nodes num: {len(extended_nodes)}")
    extended_nodes = sorted(extended_nodes)
    for n_i in range(len(extended_nodes)):
        print(extended_nodes[n_i])
    lang.clauses += extended_nodes


def eval_task(args, lang, FC, objs):
    NSFR = nsfr.get_nsfr_model(args, lang, FC, lang.clauses)
    target_preds = list(set([c.head.pred.name for c in lang.clauses]))
    # clause evaluation
    ils, dls = evaluation(args, NSFR, target_preds, objs)
    return ils


def remove_trivial_atoms(args, lang, FC, clauses, objs, data):
    lang.trivial_atom_terms = []
    # clause extension
    clauses = extension(args, lang, clauses)
    NSFR = nsfr.get_nsfr_model(args, lang, FC, clauses)
    target_preds = list(set([c.head.pred.name for c in clauses]))
    # clause evaluation
    img_scores, clause_scores = evaluation(args, NSFR, target_preds, objs, data)
    trivial_c = [clauses[i] for i in range(len(clause_scores)) if clause_scores[i] < args.th_inv_nc]
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


def alpha(args, ocm):
    obj_num = 1
    lang = init_ilp(args, obj_num)
    lang.reset_lang(g_num=1)
    VM = valuation.get_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM)
    C = lang.load_init_clauses()
    for g_i in range(len(ocm)):
        lang.reset_lang(g_num=1)
        df_search(args, lang, C, FC, ocm[g_i:g_i + 1])
        lang.variable_set_id(g_i)
        name_dict = llama_call.rename_predicates(lang)
        lang.record_milestone()
    lang.clear_repeat_language()
    return lang


def alpha_test(args, ocm, lang):
    VM = valuation.get_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM, given_attrs=lang.attrs)
    pred = eval_task(args, lang, FC, ocm).squeeze()
    return pred
