# Created by shaji at 24/06/2024
import torch
import datetime

import config
from src.utils import log_utils

from src.alpha.fol.Language import Language
from src.alpha import valuation, facts_converter, nsfr


def init_ilp(args):
    lang = Language(args)
    return lang


def extension(args, lang, clauses):
    refs = []
    B_ = []

    refinement_generator = RefinementGenerator(lang=lang)
    for c in clauses:
        refs_i = refinement_generator.refinement_clause(c)
        unused_args, used_args = log_utils.get_unused_args(c)
        refs_i_removed = remove_duplicate_clauses(refs_i, unused_args, used_args, args)
        # remove already appeared refs
        refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
        B_.extend(refs_i_removed)
        refs.extend(refs_i_removed)

    # remove semantic conflict clauses
    refs_no_conflict = remove_conflict_clauses(refs, lang.pi_clauses, args)
    if len(refs_no_conflict) == 0:
        refs_no_conflict = clauses
        args.is_done = True

    if args.show_process:
        log_utils.add_lines(f"=============== extended clauses =================", args.log_file)
        for ref in refs_no_conflict:
            log_utils.add_lines(f"{ref}", args.log_file)
    return refs_no_conflict

def eval_ims(NSFR, args, pred_names, pos_group_pred=None, batch_size=None):
    """ input: clause, output: score """
    bz = args.bs_clause_eval
    data_size = len(pos_group_pred)
    V_T_pos = torch.zeros(len(NSFR.clauses), data_size, len(NSFR.atoms)).to(args.device)
    for i in range(int(data_size / batch_size)):
        g_tensors_pos = pos_group_pred[i * bz:(i + 1) * bz]
        V_T_pos[:, i * bz:(i + 1) * bz, :] = NSFR.clause_eval_quick(g_tensors_pos)
    # each score needs an explanation
    score_positive = NSFR.get_target_prediciton(V_T_pos, pred_names, args.device)
    if score_positive.size(2) > 1:
        score_positive = score_positive.max(dim=2, keepdim=True)[0]
    ims = score_positive[:, :, 0]

    return ims


def sort_clauses_by_score(clauses, scores_all, scores):
    clause_with_scores = []
    for c_i, clause in enumerate(clauses):
        score = scores[:, c_i]
        clause_with_scores.append((clause, score, scores_all[c_i]))

    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][2], reverse=True)
        return c_sorted

    return clause_with_scores


def evaluation(args, NSFR, target_preds, eval_data=None):
    # image level scores
    ims  = eval_ims(NSFR, args, target_preds)
    # dataset level scores
    dms = ims.sum(dim=1) / ims.shape[1]
    return ims, dms


def beam_search(args, lang, C, FC):
    """
        given one or multiple neural predicates, searching for high scoring clauses, which includes following steps
        1. extend given initial clauses
        2. evaluate each clause
        3. prune clauses

    """
    # eval_pred = ['kp']
    search_step = 0
    clause_with_scores = []
    clauses = C
    while search_step <= args.max_bs_step:
        # clause extension
        extended_C = extension(args, lang, clauses)
        if args.is_done: break

        NSFR = nsfr.get_nsfr_model(args, lang, FC, extended_C)
        target_preds = [extended_C[0].head.pred.name]
        # clause evaluation
        img_scores, clause_scores = evaluation(args, NSFR, target_preds)
        # classify clauses
        clause_with_scores = sort_clauses_by_score(clauses, img_scores, clause_scores)

        # prune clauses
        if args.pi_top > 0:
            clauses, clause_with_scores = prune_clauses(clause_with_scores, args)
        else:
            clauses = logic_utils.top_select(clause_with_scores, args)

        # save data
        lang.all_clauses += clause_with_scores
        search_step += 1

    if len(clauses) > 0:
        lang.clause_with_scores = clause_with_scores

    # lang.clauses = args.last_refs
    check_result(args, clause_with_scores)

    return clauses, pred_related_data


def alpha(args, data):
    log_utils.add_lines(f"================== RUN ILP ========================", args.log_file)
    lang = init_ilp(args)
    C = lang.reset_lang(args.g_num)
    VM = valuation.get_valuation_module(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM)
    rules = None
    for i in range(args.max_step):
        args.iteration = i
        rules, step_data = beam_search(args, lang, C, FC)
        if args.is_done:
            break
    return rules
