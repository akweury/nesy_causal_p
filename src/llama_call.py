# Created by jing at 16.07.24
import requests
import json
from src.alpha.fol.logic import InvAtom, InventedPredicate

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
    prompt = (f"Rename each of the following predicates to a better name: {predicate_names}. "
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
    atom_prompt = (f"Using one or several sentences to describe a Prolog clause in English. "
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
        llama3(atom_prompt + atom_str + f". \n Only answer the converted sentences. No other words.")
        response += llama3(atom_prompt + atom_str + f". \n Only answer the converted sentences. No other words.")
    return response




def natural_rule_learner(lang):
    natural_rules = []
    for clause in lang.clauses:
        response = clause2rule(clause)
        natural_rules.append(response)
    return natural_rules

def llama_rename_predicate(sub_pred_names):
    name_str = ",".join(sub_pred_names)
    prompt = (f"The system just invented a new predicate which refers to a high-level concept that was not exist in the system language."
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
    if len(response)>30:
        response = llama3("Only return a name")
    return response


def rename_predicates(lang):
    inv_predicates = [p for p in lang.predicates if isinstance(p, InventedPredicate)]
    name_dict = {}
    for inv_p in inv_predicates:

        inv_p.old_name = inv_p.name
        sub_pred_names = [p.name for p in inv_p.sub_preds]
        inv_p.name = llama_rename_predicate(sub_pred_names)
        name_dict[inv_p.old_name] = inv_p.name
        print(f"Renaming Predicate: {inv_p.old_name} ---> {inv_p.name}")

    for c_i in range(len(lang.clauses)):
        print(f"Clause {c_i+1}/{len(lang.clauses)}: {lang.clauses[c_i]}")
    return name_dict

if __name__ == "__main__":
    response = llama3("who wrote the book godfather")
    print(response)




