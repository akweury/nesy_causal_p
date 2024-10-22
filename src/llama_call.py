# Created by jing at 16.07.24
import requests
import json

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


def clause2rule(clause):

    prompt_prefix = ("Given a set of images consist of different shapes and colors of objects. "
                     "Each image might have different number, shape, color of objects, but they do have common logic patterns."
                     "The following clause describes the common logic pattern: ")
    clause_prompt = prompt_prefix + str(clause) + " Can you convert the clause into natural language?"
    response = llama3(clause_prompt)
    return response

def natural_rule_learner(clauses):
    natural_rules = []
    for clause in clauses:
        response = clause2rule(clause)
        natural_rules.append(response)
    return natural_rules


if __name__ == "__main__":
    response = llama3("who wrote the book godfather")
    print(response)