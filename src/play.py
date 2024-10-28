# Created by jing at 22.10.24

from train_nsfr import main as clause_learner
from eval_nsfr import check_clause
from llama_call import natural_rule_learner
import config
from utils import file_utils, args_utils

# load exp arguments
args = args_utils.get_args()
# import data
train_imges = file_utils.get_all_files(config.kp_dataset / args.exp_name / "train" / "true", "png", False)[:500]

positive_images = file_utils.get_all_files(config.kp_dataset / args.exp_name / "train" / "true", "png", False)[:500]
random_imges = file_utils.get_all_files(config.kp_dataset / args.exp_name / "train" / "random", "png", False)[:500]
counterfactual_imges = file_utils.get_all_files(config.kp_dataset / args.exp_name / "train" / "counterfactual", "png",
                                                False)[:500]
# use FOL to describe the data as clauses
clauses = clause_learner(args, train_imges)


# test positive patterns
positive_acc = check_clause(args, clauses, positive_images, True)
print(f"counterfactual image accuracy: {positive_acc}")

# test counterfactual patterns
cf_acc = check_clause(args, clauses, counterfactual_imges, False)
print(f"counterfactual image accuracy: {cf_acc}")

# test random patterns
random_acc = check_clause(args, clauses, random_imges, False)
print(f"random image accuracy: {random_acc}")

# use LLM to convert clauses to natural language
rules = natural_rule_learner(clauses)
