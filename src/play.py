# Created by jing at 22.10.24
import os

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
data_folder = config.kp_dataset / args.exp_name

train_folder = data_folder / "train" / "task_true_pattern"
os.makedirs(train_folder, exist_ok=True)
test_true_folder = data_folder / "test" / "task_true_pattern"
os.makedirs(test_true_folder, exist_ok=True)
test_random_folder = data_folder / "test" / "task_random_pattern"
os.makedirs(test_random_folder, exist_ok=True)
test_cf_folder = data_folder / "test" / "task_cf_pattern"
os.makedirs(test_cf_folder, exist_ok=True)

out_train_folder = config.output / args.exp_name / "train" / "task_true_pattern"
os.makedirs(out_train_folder, exist_ok=True)
out_positive_folder = config.output / args.exp_name / "test" / "task_true_pattern"
os.makedirs(out_positive_folder, exist_ok=True)
out_random_folder = config.output / args.exp_name / "test" / "task_random_pattern"
os.makedirs(out_random_folder, exist_ok=True)
out_cf_folder = config.output / args.exp_name / "test" / "task_cf_pattern"
os.makedirs(out_cf_folder, exist_ok=True)

step_counter = 0
total_step = 8
# Generate Training Data -- Single Group Pattern
step_counter += 1
generate_training_patterns.genShapeOnShape(exp_setting["bk_groups"], 500)
print(f"Step {step_counter}/{total_step}: Generated {exp_setting['bk_groups']} training patterns")

# Generate Task Data -- Multiple Group Pattern
step_counter += 1
generate_task_patterns.genShapeOnShapeTask(exp_setting, 10)
print(f"Step {step_counter}/{total_step}: Generated {exp_setting['task_name']} task patterns")

# Import Generated Data
step_counter += 1
train_imges = file_utils.get_all_files(train_folder, "png", False)[:500]
positive_images = file_utils.get_all_files(test_true_folder, "png", False)[:500]
random_imges = file_utils.get_all_files(test_random_folder, "png", False)[:500]
counterfactual_imges = file_utils.get_all_files(test_cf_folder, "png", False)[:500]
print(f"Step {step_counter}/{total_step}: Imported training and testing data.")

# Learn Clauses from Training Data
step_counter += 1
lang = train_clauses(args, train_imges, out_train_folder)
print(f"Step {step_counter}/{total_step}: Reasoned {len(lang.clauses)} clauses")

# Test Positive Patterns
step_counter += 1

positive_acc = check_clause(args, lang, positive_images, True, out_positive_folder)
print(f"Step {step_counter}/{total_step}: Test Positive Image Accuracy: {positive_acc.mean(dim=1)}\n"
      f"{positive_acc}")

# Step 6: Test counterfactual patterns
step_counter += 1

cf_acc = check_clause(args, lang, counterfactual_imges, False, out_cf_folder)
print(f"Step {step_counter}/{total_step}: Test Counterfactual Image Accuracy: \n"
      f"{cf_acc}")

# Step 7: Test random patterns
step_counter += 1

random_acc = check_clause(args, lang, random_imges, False, out_random_folder)
print(f"Step {step_counter}/{total_step}: random image accuracy: {random_acc}")

# Step 8: Using LLM to convert clauses to natural language
step_counter += 1

rules = natural_rule_learner(lang)
print(f"Step {step_counter}/{total_step}: LLM Conversion.")
