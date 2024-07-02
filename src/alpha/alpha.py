# Created by shaji at 24/06/2024
import torch
import datetime

import config
from src.utils import log_utils

from . import valuation, facts_converter, nsfr, pruning
from .fol import language
from .fol import refinement


def init_ilp(args):
    lang = language.Language(args)
    return lang


def extension(args, lang, clauses):
    refs = []
    B_ = []

    refinement_generator = refinement.RefinementGenerator(lang=lang)
    for c in clauses:
        refs_i = refinement_generator.refinement_clause(c)
        unused_args, used_args = language.get_unused_args(c)
        refs_i_removed = pruning.remove_duplicate_clauses(refs_i, unused_args, used_args, args)
        # remove already appeared refs
        refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
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
    return refs


def eval_ims(NSFR, args, pred_names):
    """ input: clause, output: score """
    data = NSFR.fc.vm.dataset
    atom_values = NSFR.clause_eval_quick(data)
    # each score needs an explanation
    score_positive = NSFR.get_target_prediciton(atom_values, pred_names, args.device)
    # if score_positive.size(2) > 1:
    #     score_positive = score_positive.max(dim=2, keepdim=True)[0]

    return score_positive


def eval_dls(ils):
    dls = ils.sum(dim=1) / ils.shape[1]
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


def evaluation(args, NSFR, target_preds):
    # image level scores
    ils = eval_ims(NSFR, args, target_preds)
    # dataset level scores
    dls = eval_dls(ils)
    return ils, dls


# def pruning(C, ils, dls):
#     clause_with_scores = sort_clauses_by_score(C, ils, dls)
#     pruned_clauses = []
#
#     return pruned_clauses


def beam_search(args, lang, C, FC):
    """
        given one or multiple neural predicates, searching for high scoring clauses, which includes following steps
        1. extend given initial clauses
        2. evaluate each clause
        3. prune clauses

    """
    # eval_pred = ['kp']
    bs_step = 0
    clause_with_scores = []
    clauses = C
    for bs_step in range(args.max_bs_step):
        # clause extension
        extended_C = extension(args, lang, clauses)
        if args.is_done:
            break
        NSFR = nsfr.get_nsfr_model(args, lang, FC, extended_C)
        target_preds = list(set([c.head.pred.name for c in extended_C]))
        # clause evaluation
        ils, dls = evaluation(args, NSFR, target_preds)
        # prune clauses
        clauses = pruning.top_k_clauses(args, ils, dls, extended_C)
        # save data
        # lang.all_clauses += clause_with_scores
    # if len(clauses) > 0:
    #     lang.clause_with_scores = clause_with_scores
    # lang.clauses = args.last_refs
    # check_result(args, clause_with_scores)

    return clauses


def alpha(args, data):
    log_utils.add_lines(f"================== RUN ILP ========================", args.log_file)
    lang = init_ilp(args)
    C = lang.reset_lang(args.g_num)
    VM = valuation.get_valuation_module(args, lang, data)
    FC = facts_converter.FactsConverter(args, lang, VM)
    clauses = beam_search(args, lang, C, FC)
    return clauses
