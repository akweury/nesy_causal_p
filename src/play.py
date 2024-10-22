# Created by jing at 22.10.24



from train_nsfr import main as clause_learner
from llama_call import natural_rule_learner

# use FOL to describe the data as clauses
clauses = clause_learner()

# use LLM to convert clauses to natural language
rules = natural_rule_learner(clauses)


