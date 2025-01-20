# Created by jing at 16.07.24
import requests
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


def llama_rename_term(term_str):
    prompt = (f"Given two words to describe an object: {term_str}. "
              f"Now reorganise them and return a new phase that covers all the given descriptions but no other words."
              f"Only answer with the phase.")
    response = llama3(prompt)
    if len(response) > 30:
        response = llama3("Answered irrelevant words.")
    return response


def llama_rewrite_clause(principle, rule_level, obj_desc, group_id, group_name):
    if principle == "closure":
        prompt = (
            f"Several types of objects {obj_desc} form a shape of {group_name}, "
            f"which are considered as a group with group id {group_id}. "
            f"Now use one sentence to include the following information"
            f"1. the group name;"
            f"2. what kind of shape do the objects together form;"
            f"3. what kind of objects exist in the group {group_id} one by one, "
            f"make sure all of descriptions following same form; "
            f"describe clearly their properties, such as color, shape, etc."
            f"Describe it clear and simple. Only answer with the sentence. ")
    elif principle == "similarity":
        prompt = (
            f"According to the gestalt principle {principle}, the objects can be groups together,"
            f"the rule holds true on the {rule_level} level."
            f"For each {rule_level}, there exists object {obj_desc}."
            f"Now use one sentence to describe the rule including the following information"
            f"1. what kind of objects in the group"
            f"2. the rule holds true on the {rule_level} level."
            f"make sure all of descriptions following same form; "
            f"describe clearly their properties, such as color, shape, etc."
            f"Describe it clear and simple. Only answer with the sentence. ")

    elif principle == "proximity":
        prompt = (f"There are positive patterns and negative patterns. "
                  f"The positive patterns following the same rules, whereas the negative patterns do not. "
                  f"We found the following facts: there are always a set of objects in the pattern, "
                  f"in the positive patterns, the objects can be grouped together following the gestalt principle proximity, "
                  f"in each of the group, there exists {obj_desc}, where in the negative patterns, such rules does not hold."
                  f"Now use one sentence to describe the rule, make sure all of descriptions following same form; "
                  f"Describe it clear and simple. Only answer with the sentence. "
                  f"")

    else:
        raise ValueError
    response = llama3(prompt)
    return response


def rewrite_true_all_group_rules(rules, name_dict):
    """
    :return: natural langauge explanations of each rule
    """
    natural_language_explanations = []
    for rule in rules:
        obj_props = []
        for term in rule.body[0].terms:
            if isinstance(term, Const) and "object" in term.dtype.name:
                obj_props.append(term.name)
        if tuple(obj_props) not in name_dict:
            obj_str = ",".join(obj_props)
            obj_desc = llama_rename_term(obj_str)
            name_dict[tuple(obj_props)] = obj_desc
        else:
            obj_desc = name_dict[tuple(obj_props)]

        group_label = rule.body[0].terms[-1]
        group_id = rule.body[0].terms[-1].id
        # rewrite the whole clause
        final_clause = llama_rewrite_clause("proximity", "group", obj_desc, group_id, obj_props)
        natural_language_explanations.append(final_clause)

    return natural_language_explanations


def rewrite_true_all_image_rules(rules):
    pass


def rewrite_clauses(args, rules):
    # rephrase the terms
    name_dict = {}
    # explain different types of the rules
    final_clauses = {}
    # true all image rules
    final_clauses["true_all_groups"] = rewrite_true_all_group_rules(rules["true_all_group"], name_dict)
    final_clauses["true_all_images"] = rewrite_true_all_image_rules(rules["true_all_image"], name_dict)

    args.logger.debug(
        "\n =============== LLM: Rename terms =============== " + "".join(
            [f"\n{k} -> {v}" for k, v in name_dict.items()]))
    args.logger.debug(
        f"\n =============== LLM: Clause Description =============== " + "".join(
            [f"\n {c_i + 1}/{len(final_clauses['true_all_groups'])} {final_clauses['true_all_groups'][c_i]}" for c_i in
             range(len(final_clauses["true_all_groups"]))]))

    return final_clauses, name_dict

def rewrite_neg_clauses(args, rules):
    # rephrase the terms
    name_dict = {}
    # explain different types of the rules
    final_clauses = {}
    # true all image rules
    final_clauses["true_all_groups"] = rewrite_true_all_group_rules(rules["true_all_group"], name_dict)
    final_clauses["true_all_images"] = rewrite_true_all_image_rules(rules["true_all_image"], name_dict)

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


def convert_to_final_clauses(args, rules, test_results):
    """ explain following things using natural language:
    - what is the rule for the positive patterns
    - which rule does the negative pattern not satisfied

    return: rules in natural language, rules are not met by the negative patterns
    """
    # explain the rules in positive pattern
    llm_clauses, name_dict = rewrite_clauses(args, rules)
    # explain the reason that pattern is negative
    llm_neg_explains = rewrite_neg_clauses(args, test_results)


    natural_rules = {}
    negative_explanations = []
    return natural_rules, negative_explanations
