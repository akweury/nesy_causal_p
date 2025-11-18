import re 
from typing import List, Set
import config 

def train_dpl_baseline(final_rules, hard_facts, labels, max_epochs=10):
    dpl_rules = convert_final_rules_to_dpl_rules(final_rules)
    train_examples = prepare_train_examples(hard_facts, labels)
    arg_domains = generate_arg_domains(train_examples)
    rule_file = write_dpl_program(dpl_rules, train_examples)
    model = build_and_train_model(rule_file, final_rules, arg_domains, train_examples, max_epochs)
    scores = evaluate_image_scores(model, final_rules, train_examples, arg_domains)
    return model, scores


def generate_arg_domains(train_examples):
    img_ids = [f"img{i}" for i in range(len(train_examples))]
    # If group IDs are shared across images, you need to extract from facts
    group_ids = set()
    for example in train_examples:
        for fact in example["hard_facts"]:
            if fact == "in_group":
                num_groups = len(example["hard_facts"][fact])
                for g in range(num_groups):
                    group_ids.add(f"g{g}")
    group_ids = sorted(group_ids)

    return {
        "image_target_img": [img_ids],
        "group_target_univ": [group_ids, img_ids],
        "group_target_exist": [group_ids, img_ids],
    }

def prepare_train_examples(hard_facts_list, labels):
    """
    Create DeepProbLog-compatible examples from symbolic facts and labels.

    Args:
        hard_facts_list: List[List[str]] – one list of facts per image
        labels: List[int] – image-level labels (0 or 1)

    Returns:
        List[Dict]: list of {"hard_facts": [...], "label": int}
    """
    return [{"hard_facts": facts, "label": label} for facts, label in zip(hard_facts_list, labels)]


import re

def convert_final_rules_to_dpl_rules(final_rules):
    """
    Converts a list of final rules from your internal format into DPL-compatible rules.

    Args:
        final_rules (list): Each rule has attributes:
            - c: clause object (must have .to_string())
            - confidence: float
            - scope: str, one of {"universal", "existential", "image", None}

    Returns:
        dpl_rules (list of str): List of DPL rule strings including nn/2 and t(R)::head :- body.
    """
    dpl_rules = []

    for i, rule in enumerate(final_rules):
        rule_str = rule.c.to_string()
        # Separate head and body
        if ":-" in rule_str:
            head, body = map(str.strip, rule_str.split(":-", 1))
        else:
            head = rule_str.strip()
            body = "true"

        # Add scope suffix to predicate if applicable
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

        # Format rule for DeepProbLog
        dpl_rules.append(f'nn(rule{i}, [{rule.confidence:.6f}, {1-rule.confidence:.6f}]).')
        dpl_rules.append(f'rule{i}::{head} :- {body}.')
        
    for rule in dpl_rules:
        print(rule)  # Debug: print each rule
    return dpl_rules



def extract_predicates_from_dpl_rules(dpl_rules):
    preds = set()
    for rule in dpl_rules:
        # Skip lines like nn(rule0, [0.8, 0.2]).
        if not rule.strip().startswith("t("):
            continue

        # Remove the prefix like t(ruleX):: and split the rule into head and body
        rule_body = rule.split("::", 1)[-1].strip().rstrip(".")
        head_body_split = rule_body.split(":-")
        atoms = [head_body_split[0].strip()]
        if len(head_body_split) > 1:
            # body exists
            body = head_body_split[1].strip()
            # Safely extract atoms using regex to avoid splitting inside parentheses
            atoms.extend(re.findall(r"[a-zA-Z_]\w*\s*\([^()]*\)", body))

        for atom in atoms:
            pred_match = re.match(r"([a-zA-Z_]\w*)\s*\(", atom)
            if pred_match:
                preds.add(pred_match.group(1))

    return preds


def convert_hard_facts_to_dpl_facts(hard_facts: dict) -> list[str]:
    facts = []
    num_objs = len(hard_facts['x'])  # assuming all obj-level facts have this length

    # Object-level unary facts
    for i in range(num_objs):
        if 'has_shape' in hard_facts:
            facts.append(f"has_shape(obj{i}, {int(hard_facts['has_shape'][i])}).")
        if 'has_color' in hard_facts:
            r, g, b = hard_facts['has_color'][i].tolist()
            facts.append(f"has_color(obj{i}, [{int(r)}, {int(g)}, {int(b)}]).")
        for name in ['x', 'y', 'w', 'h']:
            if name in hard_facts:
                val = float(hard_facts[name][i])
                facts.append(f"{name}(obj{i}, {val:.4f}).")
        for key in hard_facts:
            if key.startswith("not_has_shape_"):
                shape_name = key[len("not_has_shape_"):]
                if hard_facts[key][i] > 0.5:
                    facts.append(f"not_has_shape(obj{i}, {shape_name}).")

    # Object-object binary relations (e.g., same_shape)
    for key in ['same_shape', 'same_color', 'same_size', 'mirror_x', 'same_y']:
        if key in hard_facts:
            mat = hard_facts[key]
            for i in range(num_objs):
                for j in range(num_objs):
                    if mat[i][j] > 0.5:
                        facts.append(f"{key}(obj{i}, obj{j}).")

    # Object-group relations (e.g., in_group)
    if 'in_group' in hard_facts:
        group_tensor = hard_facts['in_group']
        num_groups = group_tensor.shape[1]
        for obj_idx in range(num_objs):
            for group_idx in range(num_groups):
                if group_tensor[obj_idx][group_idx] > 0.5:
                    facts.append(f"in_group(obj{obj_idx}, grp{group_idx}).")

    # Group-level facts
    group_level_keys = ['diverse_shapes', 'unique_shapes', 'diverse_colors', 'unique_colors',
                        'diverse_sizes', 'unique_sizes', 'no_member_triangle', 'no_member_rectangle',
                        'no_member_circle']
    for key in group_level_keys:
        if key in hard_facts:
            for group_idx, val in enumerate(hard_facts[key]):
                if val > 0.5:
                    facts.append(f"{key}(grp{group_idx}).")

    # Group-size
    if 'group_size' in hard_facts:
        for group_idx, val in enumerate(hard_facts['group_size']):
            facts.append(f"group_size(grp{group_idx}, {float(val):.1f}).")

    # Image-level facts
    if 'same_group_counts' in hard_facts and hard_facts['same_group_counts'] > 0.5:
        facts.append("same_group_counts().")

    return facts


def ensure_predicates_defined(fact_strs: list[str], required_preds: set[str]) -> list[str]:
    """
    Ensure that all predicates in `required_preds` appear at least once in `fact_strs`.
    If missing, add a dummy fact like pred(dummy, dummy).
    """
    existing_preds = set()
    pred_pattern = re.compile(r"^(\w+)\s*\(")

    for fact in fact_strs:
        match = pred_pattern.match(fact)
        if match:
            existing_preds.add(match.group(1))

    missing_preds = required_preds - existing_preds
    for pred in missing_preds:
        arity = guess_predicate_arity(pred)
        dummy_args = ", ".join([f"dummy{i}" for i in range(arity)])
        fact_strs.append(f"% dummy fact for undefined predicate")
        fact_strs.append(f"{pred}({dummy_args}).")

    return fact_strs


def guess_predicate_arity(pred: str) -> int:
    """
    Estimate predicate arity based on naming convention.
    Extend this if you have a predicate registry or type info.
    """
    if pred in {"same_shape", "same_color", "same_size", "mirror_x", "same_y"}:
        return 2
    elif pred in {"in_group"}:
        return 2
    elif pred in {"has_shape", "not_has_shape", "has_color", "x", "y", "w", "h"}:
        return 2
    elif pred in {"group_size"}:
        return 2
    elif pred in {"same_group_counts"}:
        return 0
    else:
        return 1  # default: unary




def write_dpl_program(dpl_rules, train_examples):
    import tempfile
    # dpl_rules = convert_final_rules_to_dpl_rules(final_rules)
    program_parts = dpl_rules.copy()
    required_preds = extract_predicates_from_dpl_rules(dpl_rules)
    for fact_dict in train_examples:
        hard_facts = fact_dict["hard_facts"]
        fact_strs = convert_hard_facts_to_dpl_facts(hard_facts)
        fact_strs = ensure_predicates_defined(fact_strs, required_preds)
        program_parts.extend(fact_strs)

    with open(config.dpl_file_path, "w") as f:
        f.write("\n".join(program_parts))
        return f.name
    
    
from deepproblog.train import StopCondition

class FixedEpochStopCondition(StopCondition):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def is_stop(self, train_object):
        return train_object.epoch >= self.max_epochs
    
    
    
def build_and_train_model(rule_file, final_rules, arg_domains, train_examples, max_epochs):
    from deepproblog.model import Model
    from deepproblog.engines import ExactEngine
    from deepproblog.query import Query
    from deepproblog.train import train_model
    from problog.logic import Term
    import itertools

    rule_file = config.output/"test.pl"
    model = Model(str(rule_file), networks={}, load=True )
    model.set_engine(ExactEngine(model))
    for i, rule in enumerate(final_rules):
        name = f"rule{i}"
        if name in model.parameters:
            model.parameters[name].data.fill_(rule.confidence)

    queries = []
    for (_, label) in train_examples:
        for pred, domain in arg_domains.items():
            for args in itertools.product(*domain):
                term = Term(pred, *[Term(str(a)) for a in args])
                queries.append(Query(term, p=float(label)))

    train_model(model, [queries], FixedEpochStopCondition(max_epochs))
    return model


def evaluate_image_scores(model, final_rules, train_examples, arg_domains):
    from deepproblog.model import Model
    from deepproblog.engines import ExactEngine
    from deepproblog.query import Query
    from problog.logic import Term
    import itertools
    import tempfile

    dpl_rules = convert_final_rules_to_dpl_rules(final_rules)
    scores = []

    for facts, _ in train_examples:
        program = "\n".join(dpl_rules + facts)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False) as f:
            f.write(program)
            f.flush()
            eval_model = Model(f.name, networks={})
            eval_model.set_engine(ExactEngine(eval_model))

            score_sum, count = 0.0, 0
            for pred, domain in arg_domains.items():
                for args in itertools.product(*domain):
                    q = Query(Term(pred, *[Term(str(a)) for a in args]))
                    try:
                        result = eval_model.solve([q])
                        score_sum += result[q].value
                        count += 1
                    except:
                        pass
            scores.append(score_sum / count if count else 0.0)

    return scores

