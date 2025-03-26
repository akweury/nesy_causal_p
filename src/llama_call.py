# Created by X at 16.07.24
import os.path

import requests
import torch

import config
from src import bk
from src.alpha.fol.logic import InvAtom
from src.alpha.fol.logic import Const, Var

url = "http://localhost:11434/api/chat"


def llama3(prompt):
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": prompt

            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]


def query_predicate(predicates):
    old_names = [p.name for p in predicates]
    predicate_names = ",".join(old_names)
    prompt = (
        f"Rename each of the following predicates to a better name: {predicate_names}. "
        f"A pattern has groups, a group has objects. "
        f"gshape means the shape of the group"
        f"inp means in the pattern, ing means in the group,"
        f"has_color means an object has color, has shape means an object has shape."
        f"Only reply the names of each predicate, split by a comma.")
    response = llama3(prompt)
    new_names = response.split(",")
    name_dict = {old_names[i]: new_names[i] for i in range(len(old_names))}

    return name_dict


def clause2rule(clause):
    atom_prompt = (
        f"Using one or several sentences to describe a Prolog clause in English. "
        f"There are the explanations of the variables: "
        f"I, refers to an image, an image includes multiple objects, each object has its shape and color, "
        f"some of them together form another shape;"
        f"G_x, refers to a group of objects, x is the group id;"
        f"O_x, refers to an object; x is the object id."
        f"Explanations of Terms:"
        f"(color_xx, G_x, I), means an object with color_xx in the group G_x, and G_x in the image I;"
        f"(shape_xx, G_x, I), means an object with shape_xx in the group G_x, and G_x in the image I;"
        f"(group_xx, G_x, I), means a set of objects form a shape of group_xx, the group is presented as G_x, and G_x in the image I;"
        f"(O_x, G_x, I), means object O_x in the group G_x, and G_x in the image I;"
        f"(G_x, I), means G_x in the image I."
        f"\n\nThe actual group shape, object shape, object color is written in the parenthesis."
    )
    response = ""
    for atom in clause.body:
        atom_str = ""
        if isinstance(atom, InvAtom):
            for s_i, sub_pred in enumerate(atom.pred.sub_preds):
                atom_str += f"{sub_pred.name}" + f"{atom.terms[s_i]},"
        llama3(
            atom_prompt + atom_str + f". \n Only answer the converted sentences. No other words.")
        response += llama3(
            atom_prompt + atom_str + f". \n Only answer the converted sentences. No other words.")
    return response


def natural_rule_learner(lang):
    natural_rules = []
    for clause in lang.clauses:
        response = clause2rule(clause)
        natural_rules.append(response)
    return natural_rules


def llama_rename_predicate(sub_pred_names):
    name_str = ",".join(sub_pred_names)
    prompt = (
        f"The system just invented a new predicate which refers to a high-level concept that was not exist in the system language."
        f"The new predicate is usually invented based on multiple given predicates in the system language,"
        f"its functionality is the combination of the functionality of the given predicates. "
        f"For example, by combine the predicate has_shape and has_color, "
        f"where the first predicate has_shape checks if an object has specific shape,"
        f"the second predicate has_color checks if an object has specific color,"
        f"the new predicate can be invented by combining both two predicates with having the functionality "
        f"that check an object has both specific shape and specific color. "
        f"Now the question is, how to give a proper name for the new predicate. "
        f"If the given predicates are has_shape and has_color, it can be named as has_shape_and_color."
        f"Given the following given predicates, return the name of the invented predicates, the predicates are splited by comma:"
        f"{name_str}")
    response = llama3(prompt + f"\n Only answer with the name.")
    if len(response) > 30:
        response = llama3("Only return a name")
    return response


def llama_rename_term(term_str, level):
    prompt = (f"If there is an object, it has property shape and color."
              f"If there is a group, it consists with multiple objects, these objects form a shape of group, "
              f"the group can also has object number as its property."
              f"Now, there is a {level}. It has properties {term_str}."
              f"Now reorganise them and return a new phase that covers all the given descriptions but no other words. "
              f"Do not omit any number if there exist such. Only answer with the phase. No other words.")
    response = llama3(prompt)
    if len(response) > 50:
        response = llama3("Answered irrelevant words.")
    return response


def llama_rewrite_clause(principle, rule_level, facts, obj_num):
    intro = (f"Rewrite the following sentences and make it sounds naturally."
             f"There's a consistent pattern in all the positive images.")
    end = "Only answer with the rewrite sentences. No other sentence."
    if principle == "closure":
        prompt = (f"When grouping the objects in the positive image via gestalt principle closure, "
                  f"they following the following fact: {facts}."
                  f"where in the negative patterns, such rules does not hold."
                  )
    elif principle == "similarity_color":
        prompt = (
            f"The objects are grouped based on shared attributes color, forming distinct clusters."
            f"For one group, it includes {facts}."
        )
    elif principle == "similarity_shape":
        prompt = (
            f"The objects are grouped based on shared attributes shape, forming distinct clusters."
            f"For one group, it includes {facts}."
        )
    elif principle == "proximity":
        prompt = (f"When grouping the objects in the positive image via gestalt principle proximity, "
                  f"there exists at least {obj_num} {facts} in every {rule_level}, "
                  f"where in the negative patterns, such rules does not hold.")
    else:
        raise ValueError
    response = llama3(intro + prompt + end)
    return response


def rewrite_true_all_group_rules(rule, natural_language_explanations):
    """
    :return: natural langauge explanations of each rule
    """
    # facts = []
    # obj_props = []
    # if rule_type in ["true_all_group"]:
    #     level = "object"
    # else:
    #     raise ValueError
    # for term in rule.body[0].terms:
    #     if isinstance(term, Const) and "object" in term.dtype.name:
    #         obj_props.append(term.name)
    # if tuple(obj_props) not in name_dict:
    #     obj_str = ",".join(obj_props)
    #     obj_desc = llama_rename_term(obj_str, level)
    #     name_dict[tuple(obj_props)] = obj_desc

    prompt = (f"The natural language converter M_nlc maps the logical rules R into a human readable textual description T. "
              f"We define:  T = M_nlc(R) where T is a sequence of tokens (t_1, t_2,..., t_m) in a target language (e.g. English). "
              f"This mapping often relies on predefined templates or trained language-generation models "
              f"that interpret logical constructs and translate them into coherent sentences. "
              f"For example, if r_i is a constraint stating "
              f"img(I):-exist_obj_shape(circle,O_0,G_0,I),in_group(O_0,G_0,I),in_pattern(G_0,I)."
              f"the converter might produce a sentence such as :"
              f"There is a group (G_0) of objects in the pattern (I), "
              f"the group (G_0) contains object (O_0) with shape of circle (properties)."
              f"Thus, the natural language converter bridges the gap between symbolic logic rules and human-friendly explanations, "
              f"enabling users to readily understand the system's inferred patterns or constraints."
              f"More examples:"
              f"if r_i is a constraint stating "
              f"target(I):-exist_obj_color_exist_obj_shape(purple,square,O_1,G_0),in_group(O_1,G_0,I),in_pattern(G_0,I).,"
              f"the converter might produce a sentence such as :"
              f"There is a group (G_0) of objects in the pattern (I), "
              f"the group (G_0) contains object (O_0), which is a purple square (properties)."
              f"Now given a clause {str(rule)}, convert it to natural language. "
              f"Only answer with the converted sentences."
              )

    final_clause = llama3(prompt)
    natural_language_explanations["true_all_group"][rule] = final_clause

    return natural_language_explanations


def rewrite_true_all_image_rules(rule, name_dict, rule_explanations, principle, obj_num, level):
    if rule in rule_explanations["true_all_image"]:
        return rule_explanations
    facts = []
    obj_props = []
    for term in rule.body[0].terms:
        if isinstance(term, Const) and "object" in term.dtype.name:
            obj_props.append(term.name)
    if tuple(obj_props) not in name_dict:
        obj_str = ",".join(obj_props)
        obj_desc = llama_rename_term(obj_str, level)
        name_dict[tuple(obj_props)] = obj_desc
    else:
        obj_desc = name_dict[tuple(obj_props)]
    facts.append(obj_desc)
    # rewrite the whole clause
    final_clause = llama_rewrite_clause(principle, "image", obj_desc, obj_num)
    rule_explanations["true_all_image"][rule] = final_clause

    return rule_explanations


def load_llm_responses(task_id):
    explanation_file = config.models / f"explanations_{task_id}.pt"
    if os.path.exists(explanation_file):
        rule_explanations = torch.load(explanation_file)["rule_explanations"]
        name_dict = torch.load(explanation_file)["name_dict"]
        if "true_all_group" not in rule_explanations:
            rule_explanations["true_all_group"] = []
        if "true_all_image" not in rule_explanations:
            rule_explanations["true_all_image"] = []
        if "true_exact_one_group" not in rule_explanations:
            rule_explanations["true_exact_one_group"] = []
    else:
        rule_explanations = {"true_all_group": {},
                             "true_all_image": {},
                             "true_exact_one_group": {}}
        name_dict = {}
    return rule_explanations, name_dict


def save_llm_responses(rule_explanations, name_dict, task_id):
    explanation_file = config.models / f"explanations_{task_id}.pt"
    data = {"rule_explanations": rule_explanations,
            "name_dict": name_dict}
    torch.save(data, explanation_file)


def rewrite_clauses(args, rules, principle, task_id):
    rule_explanations, name_dict = load_llm_responses(task_id)
    # true all image rules
    for rule in rules:
        rewrite_true_all_group_rules(rule["rule"], rule_explanations)
    args.logger.debug(
        "\n =============== LLM: Rename terms =============== " + "".join(
            [f"\n{k} -> {v}" for k, v in name_dict.items()]))
    save_llm_responses(rule_explanations, name_dict, task_id)
    return rule_explanations, name_dict


def llama_negative_reason(ocm, rule):
    return response


def rewrite_false_all_group_examples(rule, negative_data):
    negative_details, negative_groups = negative_data["negative_details"], negative_data["negative_groups"]
    principle = bk.gestalt_principles[negative_data["principle"]]

    negative_responses = []
    for e_i in range(len(negative_details)):
        example_response = []
        example_obj_num = sum([len(ig["ocm"]) for igs in negative_groups for ig in igs])
        group_num = len(negative_groups[e_i])
        for g_i in range(len(negative_details[e_i])):
            if not negative_details[e_i][g_i]:
                # add one prompt
                ocm = negative_groups[e_i][g_i]["ocm"]
                gcm = negative_groups[e_i][g_i]["gcm"]
                rule_text = list(rule["true_all_group"].values())[0]
                obj_dicts = [bk.tensor2dict(o) for o in ocm]
                positions = [(obj["position"] * 512).int().tolist() for obj in obj_dicts]
                group_pos = (gcm[:, : 2] * 512).int().tolist()
                shapes = [bk.bk_shapes[1:][obj['shape'].argmax()] for obj in obj_dicts]
                colors = [bk.color_dict_rgb2name[tuple((obj['color'] * 255).int().tolist())] for obj in obj_dicts]

                # prompt variables
                obj_text = ", ".join([f"{colors[i]} {shapes[i]}" for i in range(len(obj_dicts))])

                prompt = (f"This is a negative image, because not all the group follows the rule. {rule_text} "
                          f"In this negative image, the group consists by objects {obj_text} at position {group_pos} "
                          f"does not follow the rule. To make it positive, one action can be performed by "
                          f"change the property from xx to xxx.")
                neg_response = llama3(prompt)
                example_response.append(neg_response)
        negative_responses.append(example_response)
    return negative_responses


def rewrite_neg_clauses(args, rule, negative_data):
    # rephrase the terms
    name_dict = {}
    # explain different types of the rules
    final_clauses = {}
    # true all image rules
    # rewrite_false_all_group_examples(rule, negative_data)

    args.logger.debug(
        "\n =============== LLM: Rename terms =============== " + "".join(
            [f"\n{k} -> {v}" for k, v in name_dict.items()]))
    args.logger.debug(
        f"\n =============== LLM: Clause Description =============== " + "".join(
            [f"\n {c_i + 1}/{len(final_clauses['true_all_groups'])} {final_clauses['true_all_groups'][c_i]}" for c_i in
             range(len(final_clauses["true_all_groups"]))]))

    return final_clauses, name_dict


if __name__ == "__main__":
    response = llama3("who wrote the book godfather")
    print(response)


def convert_to_final_clauses(args, rules, test_results, principle, task_id):
    """ explain following things using natural language:
    - what is the rule for the positive patterns
    - which rule does the negative pattern not satisfied

    return: rules in natural language, rules are not met by the negative patterns
    """
    # explain the rules in positive pattern
    llm_clauses, name_dict = rewrite_clauses(args, rules, principle, task_id)

    # explain the reason that pattern is negative
    # llm_neg_explains = rewrite_neg_clauses(args, llm_clauses, test_results)

    return llm_clauses
