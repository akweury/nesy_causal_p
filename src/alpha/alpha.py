# Created by shaji at 24/06/2024
import torch
import datetime

import config
from src.utils import log_utils

from . import valuation, facts_converter, nsfr, pruning
from .fol import language
from .fol import refinement


def init_ilp(args, variable_num):
    lang = language.Language(args.variable_symbol, variable_num, args.lark_path,
                             args.fm_num, args.phi_num, args.rho_num)
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


def evaluation(args, NSFR, target_preds):
    # image level scores
    ils = eval_ims(NSFR, args, target_preds)
    # dataset level scores
    dls = eval_dls(ils)
    return ils, dls


def beam_search(args, lang, C, FC):
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
        pruned_c = pruning.top_k_clauses(args, ils, dls, extended_C)
        # save data
        if len(pruned_c) == 0:
            break
        else:
            clauses = pruned_c

    return clauses


def alpha(args, fms, images):
    for obj_num in range(1, args.max_obj_num):
        args.fm_num = len(fms)
        lang = init_ilp(args, obj_num)
        C = lang.reset_lang()
        VM = valuation.get_valuation_module(args, lang)
        FC = facts_converter.FactsConverter(args, lang, VM)
        clauses = beam_search(args, lang, C, FC)
    return clauses
