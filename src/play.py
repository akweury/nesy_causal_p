# Created by jing at 22.10.24
import config

from train_nsfr import train_clauses
from eval_nsfr import check_clause
from llama_call import natural_rule_learner
from utils import file_utils, args_utils
from kandinsky_generator import generate_training_patterns, generate_task_patterns
from src.alpha.fol import bk

# load exp arguments
args = args_utils.get_args()
exp_setting = bk.exp_demo

step_counter = 0
total_step = 8
# Step 1: Generate Training Data -- Single Group Pattern
step_counter += 1
generate_training_patterns.genShapeOnShape(exp_setting["bk_groups"], 500)
print(f"Step {step_counter}/{total_step}: Generated {exp_setting['bk_groups']} training patterns")

# Step 2: Generate Task Data -- Multiple Group Pattern
step_counter += 1
generate_task_patterns.genShapeOnShapeTask(exp_setting, 10)
print(f"Step {step_counter}/{total_step}: Generated {exp_setting['task_name']} task patterns")

# Step 3: Import Generated Data
step_counter += 1

exp_folder = config.kp_dataset / args.exp_name
train_imges = file_utils.get_all_files(exp_folder / "train" / "task_true_pattern", "png", False)[:500]
positive_images = file_utils.get_all_files(exp_folder / "train" / "task_true_pattern", "png", False)[:500]
random_imges = file_utils.get_all_files(exp_folder / "test" / "task_random_pattern", "png", False)[:500]
counterfactual_imges = file_utils.get_all_files(exp_folder / "test" / "task_cf_pattern", "png", False)[:500]
print(f"Step {step_counter}/{total_step}: Imported training and testing data.")

# Step 4: Learn Clauses from Training Data
step_counter += 1

lang = train_clauses(args, train_imges)
print(f"Step {step_counter}/{total_step}: Reasoned {len(lang.clauses)} clauses")

# Step 5: Test Positive Patterns
step_counter += 1

positive_acc = check_clause(args, lang, positive_images, True)
print(f"Step {step_counter}/{total_step}: Test Positive Image Accuracy: {positive_acc}")

# Step 6: Test counterfactual patterns
step_counter += 1

cf_acc = check_clause(args, lang, counterfactual_imges, False)
print(f"Step {step_counter}/{total_step}: Test Counterfactual Image Accuracy: {cf_acc}")

# Step 7: Test random patterns
step_counter += 1

random_acc = check_clause(args, clauses, random_imges, False)
print(f"Step {step_counter}/{total_step}: random image accuracy: {random_acc}")

# Step 8: Using LLM to convert clauses to natural language
step_counter += 1

rules = natural_rule_learner(clauses)
