# Created by jing at 16.07.24
import requests
import json
from src.alpha.fol.logic import InvAtom

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
    atom_prompt = (f"Convert the logic atom to natural language sentence using the following explanations: "
                   f"group shape means a group of objects form a shape of XXX, "
                   f"has shape means exist an object in the group has shape XXX,"
                   f"has color means exist an object in the group has color XXX,"
                   f"the actual group shape, object shape, object color is written in the parenthesis."
                   f"The atom is ")
    response = ""
    for atom in clause.body:
        atom_str = ""
        if isinstance(atom, InvAtom):
            for s_i, sub_pred in enumerate(atom.pred.sub_preds):
                atom_str += f"{sub_pred.name}" + f"{atom.terms[s_i]},"

        response += llama3(atom_prompt + atom_str + f"\n Only return the converted result, no extra words.")
    return response


def natural_rule_learner(lang):
    natural_rules = []
    for clause in lang.clauses:
        response = clause2rule(clause)
        natural_rules.append(response)
    return natural_rules


if __name__ == "__main__":
    response = llama3("who wrote the book godfather")
    print(response)
