import re
import tempfile
from typing import List
# from mbg.scorer.calibrator import DeepProbLogCalibrator, InMemoryDataset, FixedEpochStopCondition


# from deepproblog.query import Term, Query
# from deepproblog.model import Model
# from deepproblog.engines import ExactEngine
# from problog.logic import Term
# from deepproblog.train import train_model


def convert_final_rules_to_dpl_rules(final_rules):
    dpl_rules = []
    for i, rule in enumerate(final_rules):
        rule_str = rule.c.to_string().strip()
        if ":-" in rule_str:
            head, body = rule_str.split(":-", 1)
            body = body.strip()
        else:
            head = rule_str
            body = "true"
        head = head.strip()

        # Adjust head predicate according to rule.scope
        scope = getattr(rule, "scope", None)
        if scope:
            match = re.match(r"(\w+)\((.*)\)", head)
            if match:
                pred, args = match.groups()
                if scope == "existential":
                    pred += "_exist"
                elif scope == "universal":
                    pred += "_univ"
                elif scope == "image":
                    pred += "_img"
                head = f"{pred}({args})"

        # Set learnable rule with initial confidence from rule.confidence (default 0.5)
        conf = getattr(rule, "confidence", 0.5)
        dpl_rules.append(f"nn('rule{i}', [{conf:.4f}, 1.0]).")
        dpl_rules.append(f"t('rule{i}')::{head} :- {body}.")

    return dpl_rules


def save_rules_to_tempfile(rule_strings: List[str]) -> str:
    temp_file = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".pl", delete=False)
    temp_file.write("\n".join(rule_strings))
    temp_file.flush()
    return temp_file.name

#
# def convert_hard_facts_to_dpl_facts(hard: dict) -> list[str]:
#     facts = []
#     obj_num = hard["has_shape"].shape[0]
#     group_num = hard["in_group"].shape[1] if "in_group" in hard else 0
#
#     # Object-level facts
#     for i in range(obj_num):
#         facts.append(f"has_shape(o{i}, {int(hard['has_shape'][i])}).")
#         r, g, b = map(int, hard["has_color"][i].tolist())
#         facts.append(f"has_color(o{i}, rgb_{r}_{g}_{b}).")
#
#     # Group membership
#     for i in range(obj_num):
#         for j in range(group_num):
#             if hard["in_group"][i][j] == 1.0:
#                 facts.append(f"in_group(o{i}, g{j}).")
#
#     # Binary object-object relations
#     binary_preds = ["same_shape", "same_color",
#                     "same_size", "mirror_x", "same_y"]
#     for pred in binary_preds:
#         if pred in hard:
#             mat = hard[pred]
#             for i in range(obj_num):
#                 for j in range(obj_num):
#                     if i != j and mat[i][j] == 1.0:
#                         facts.append(f"{pred}(o{i}, o{j}).")
#
#     # Group-level properties (boolean per group)
#     group_preds = ["diverse_shapes", "unique_shapes", "diverse_colors", "unique_colors",
#                    "diverse_sizes", "unique_sizes", "no_member_triangle", "no_member_rectangle",
#                    "no_member_circle"]
#     for pred in group_preds:
#         if pred in hard:
#             for g in range(group_num):
#                 if hard[pred][g] == 1.0:
#                     facts.append(f"{pred}(g{g}).")
#
#     # Scalar facts
#     if "same_group_counts" in hard:
#         facts.append(
#             f"same_group_counts({int(hard['same_group_counts'].item())}).")
#
#     return facts


# def extract_predicates_from_final_rules(final_rules) -> set[str]:
#     """
#     Extracts all predicate names used in the body of ScoredRule clauses.
#
#     Args:
#         final_rules: A list of ScoredRule objects, each with a .c (clause string).
#
#     Returns:
#         A set of unique predicate names (e.g., 'has_shape', 'same_color').
#     """
#     predicates = set()
#     for rule in final_rules:
#         if ":-" in rule.c.to_string():
#             body = rule.c.to_string().split(":-", 1)[1]
#         else:
#             body = rule.c.to_string()
#         matches = re.findall(r'(\w+)\s*\(', body)
#         predicates.update(matches)
#     return predicates


# def ensure_predicates_defined(facts: list[str], required_predicates: set[str]) -> list[str]:
#     """
#     Ensures that all predicates required by the rule set are defined in the facts list.
#     If a required predicate is missing, a dummy fact is added to avoid DeepProbLog errors.
#
#     Args:
#         facts: A list of existing fact strings for one image (e.g., ['has_shape(o0, 0).', ...]).
#         required_predicates: A set of predicate names used in the rule body (e.g., {'has_shape', 'same_color'}).
#
#     Returns:
#         An updated list of fact strings that includes dummy facts for all required predicates.
#     """
#     existing_predicates = {fact.split('(')[0] for fact in facts}
#     missing_predicates = required_predicates - existing_predicates
#
#     for pred in missing_predicates:
#         # Default dummy fact with two generic arguments
#         facts.append(f"{pred}(dummy0, dummy1).")
#
#     return facts


#
# def train_dpl_baseline(final_rules, train_examples, arg_domains, max_epochs=10):
#     rule_file = write_dpl_program(final_rules, train_examples)
#     model = build_and_train_model(rule_file, final_rules, arg_domains, train_examples, max_epochs)
#     scores = evaluate_image_scores(model, final_rules, train_examples, arg_domains)
#     return model, scores
#


#
# def train_dpl(final_rules, hard_list, obj_list, group_list, img_labels, arg_domains_dict):
#     import itertools
#     from deepproblog.query import Query
#     from problog.logic import Term
#
#     dpl_rule_templates = convert_final_rules_to_dpl_rules(final_rules)
#     program_parts = dpl_rule_templates.copy()
#     required_preds = extract_predicates_from_final_rules(final_rules)
#
#     all_queries = []
#
#     for i, (hard_facts, label) in enumerate(zip(hard_list, img_labels)):
#         facts = convert_hard_facts_to_dpl_facts(hard_facts)
#         facts = ensure_predicates_defined(facts, required_preds)
#         program_parts.extend(facts)
#
#         # Construct queries for this image
#         for pred in arg_domains_dict:
#             arg_domains = arg_domains_dict[pred]
#             for args in itertools.product(*arg_domains):
#                 query_term = Term(pred, *[Term(str(a)) for a in args])
#                 query = Query(query_term, p=float(label))
#                 all_queries.append(query)
#
#     # Write all rules and facts into a single program file
#     program_str = "\n".join(program_parts)
#     with tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False) as temp_file:
#         temp_file.write(program_str)
#         temp_file.flush()
#         rule_file_path = temp_file.name
#
#     model = Model(rule_file_path, networks={})
#     model.set_engine(ExactEngine(model))
#
#     # Initialize parameters from final_rules
#     for i, rule in enumerate(final_rules):
#         param_name = f"rule{i}"
#         if param_name in model.parameters:
#             model.parameters[param_name].data.fill_(rule.confidence)
#
#     stop_condition = FixedEpochStopCondition(max_epochs=10)
#     train_model(model, [all_queries], stop_condition=stop_condition)
#
#     # Save calibrated rule program
#     calibrated_rule_file = rule_file_path.replace(".pl", "_calibrated.pl")
#     with open(calibrated_rule_file, "w") as f:
#         f.write(program_str + "\n")
#
#     # Print confidence updates
#     print("\n=== Rule Confidence Update ===")
#     for i, rule in enumerate(final_rules):
#         param_name = f"rule{i}"
#         if param_name in model.parameters:
#             new_conf = model.parameters[param_name].item()
#             old_conf = rule.confidence
#             diff = new_conf - old_conf
#             rule_str = rule.c.to_string().strip()
#             scope = getattr(rule, "scope", "N/A")
#             print(f"Rule {i} (scope={scope}):\n  {rule_str}\n  Old: {old_conf:.3f}, New: {new_conf:.3f}, Δ: {diff:+.3f}")
#
#     return model
#
#
# def test_dpl(model, final_rules, test_hard_facts):
#     """
#     Use the trained DeepProbLog model to predict the label for a test image.
#
#     Args:
#         model: The trained DeepProbLog Model (from train_dpl).
#         final_rules: The rules used for training (needed to reconstruct the logic program).
#         test_hard_facts: The hard facts for the test image (dict, same format as training).
#
#     Returns:
#         The result of model.solve([query]), which contains the prediction.
#     """
#     # Prepare rules and facts for the test image
#     dpl_rule_templates = convert_final_rules_to_dpl_rules(final_rules)
#     rule_template_str = "\n".join(dpl_rule_templates)
#     required_preds = extract_predicates_from_final_rules(final_rules)
#     facts = convert_hard_facts_to_dpl_facts(test_hard_facts)
#     facts = ensure_predicates_defined(facts, required_preds)
#     program_str = rule_template_str + "\n" + "\n".join(facts)
#
#     # Write combined program to a temporary file
#     import tempfile
#     with tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False) as temp_file:
#         temp_file.write(program_str)
#         temp_file.flush()
#         rule_file_path = temp_file.name
#
#     # Create a new model for this test example
#     from deepproblog.model import Model
#     from deepproblog.engines import ExactEngine
#     model = Model(rule_file_path, networks={})
#     model.set_engine(ExactEngine(model))
#
#     # Query with variable label
#     from problog.logic import Var
#     Y = Var("Y")
#     query_term = Term("label", Term("img"), Y)
#     query = Query(query_term)
#
#     result = model.solve([query])
#     return result
#
#
# def test_dpl_any_head(model, final_rules, test_hard_facts, arg_domains_dict):
#     """
#     Predicts the image score by checking if any head predicate in the final rules is true.
#
#     Args:
#         model: The trained DeepProbLog Model (from train_dpl).
#         final_rules: The rules used for training.
#         test_hard_facts: The hard facts for the test image.
#         arg_domains_dict: Dict mapping predicate name to list of argument domains (e.g., {'group_target': [[g0, g1], ['positive']]}).
#
#     Returns:
#         1.0 if any head predicate instance is true (prob > 0), else 0.0.
#     """
#     import itertools
#
#     # Prepare rules and facts for the test image
#     dpl_rule_templates = convert_final_rules_to_dpl_rules(final_rules)
#     rule_template_str = "\n".join(dpl_rule_templates)
#     required_preds = extract_predicates_from_final_rules(final_rules)
#     facts = convert_hard_facts_to_dpl_facts(test_hard_facts)
#     facts = ensure_predicates_defined(facts, required_preds)
#     program_str = rule_template_str + "\n" + "\n".join(facts)
#
#     # Write combined program to a temporary file
#     import tempfile
#     with tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False) as temp_file:
#         temp_file.write(program_str)
#         temp_file.flush()
#         rule_file_path = temp_file.name
#
#     # Create a new model for this test example
#     from deepproblog.model import Model
#     from deepproblog.engines import ExactEngine
#     model = Model(rule_file_path, networks={})
#     model.set_engine(ExactEngine(model))
#
#     from deepproblog.query import Query
#     from problog.logic import Term
#
#     # Extract all unique head predicates from final_rules
#     head_predicates = set()
#     for rule in final_rules:
#         rule_str = rule.c.to_string()
#         head = rule_str.split(":-")[0].strip()
#         pred_name = head.split("(")[0]
#         head_predicates.add(pred_name)
#
#     # For each head predicate, check all possible groundings
#     for pred in head_predicates:
#         if pred not in arg_domains_dict:
#             continue  # skip if no domain info
#         arg_domains = arg_domains_dict[pred]
#         for args in itertools.product(*arg_domains):
#             query_term = Term(pred, *[Term(str(a)) for a in args])
#             query = Query(query_term)
#             result = model.solve([query])
#             if result and hasattr(result[0], "value") and result[0].value > 0:
#                 return 1.0  # Image is positive if any head predicate instance is true
#     return 0.0  # Otherwise, image is negative
#
#
# def get_dpl_args(final_rules, group_list):
#     head_predicates = set()
#     for rule in final_rules:
#         rule_str = rule.c.to_string()
#         head = rule_str.split(":-")[0].strip()
#         match = re.match(r"(\w+)\((.*)\)", head)
#         if match:
#             pred, _ = match.groups()
#             scope = getattr(rule, "scope", None)
#             if scope == "existential":
#                 pred += "_exist"
#             elif scope == "universal":
#                 pred += "_univ"
#             elif scope == "image":
#                 pred += "_img"
#             head_predicates.add(pred)
#
#     group_names = [f"g{g['id']}" for g in group_list]
#
#     arg_domains_dict = {}
#     for pred in head_predicates:
#         if "group_target" in pred:
#             arg_domains_dict[pred] = [group_names, ["positive"]]
#         elif "image_target" in pred:
#             arg_domains_dict[pred] = [["positive"]]
#         # Add more as needed for your use case
#
#     return arg_domains_dict
