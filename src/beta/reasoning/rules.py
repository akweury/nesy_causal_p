# Created by jing at 12.03.25

from src.beta.reasoning.lang import language as lang
from src.beta.reasoning.lang import primitive
import re


def reasoning_object_relations(args, all_matrix, mode):
    """
    Process object relations by reasoning over the input matrix.

    This function initializes the language for object-level reasoning, sets up a valuation model
    and fact converter, then iterates through each example and group to extract and count object
    clauses. It aggregates unique clauses from all groups, updates the language configuration,
    and saves the updated language state.

    Args:
        args: Configuration and runtime arguments.
        all_matrix (list): List of examples, where each example is a list of groups containing 'ocm'.
        mode (str): Mode identifier used when saving the language.

    Returns:
        lang: The updated language object.
    """
    # Initialize language and reset to object-level.
    lang = Lang()
    lang.reset_lang(g_num=1, level="object")

    # Set up the valuation model and fact converter.
    VM = primitive.get_primitive_predicates(args, lang)
    FC = facts_converter.FactsConverter(args, lang, VM)
    initial_clauses = lang.load_obj_init_clauses()

    all_groups = []
    # Iterate over each example.
    for example in all_matrix:
        example_groups = []
        # Process each group in the current example.
        for group in example:
            ocms = group["ocm"]
            obj_clauses = {}
            # Process each object candidate in the group.
            for o_i, ocm in enumerate(ocms):
                clauses = df_search(args, lang, initial_clauses, FC, ocm.unsqueeze(0), 1, level="object")
                if not clauses:
                    continue
                for clause in clauses:
                    # Adjust the clause's object ID.
                    clause = clause_op.change_clause_obj_id(clause, args, o_i, bk.variable_symbol_obj)
                    obj_clauses[clause] = obj_clauses.get(clause, 0) + 1
            example_groups.append({"obj_clauses": obj_clauses})
        all_groups.append(example_groups)
    lang.all_groups = all_groups

    # Aggregate unique clauses from all groups.
    unique_clauses = set()
    for example_groups in all_groups:
        for group_data in example_groups:
            unique_clauses.update(group_data["obj_clauses"].keys())
    o_clauses = list(unique_clauses)

    # Update language with the discovered clauses.
    lang.update_consts(o_clauses)
    lang.generate_atoms(o_clauses)
    lang.update_predicates(o_clauses)
    lang.clauses = o_clauses

    save_lang(args, lang, mode, "object")
    return lang


def reasoning_group_relations(args, all_matrix, mode):
    example_num = len(all_matrix)
    all_clauses = []
    for example_i in range(example_num):
        group_num = len(all_matrix[0])
        lang = init_ilp(args, group_num)
        lang.reset_lang(g_num=group_num, level="group")
        VM = valuation.get_group_valuation_module(args, lang)
        FC = facts_converter.FactsConverter(args, lang, VM)
        C = lang.load_group_init_clauses()
        example_clauses = []
        example_groups = all_matrix[example_i]
        enum_example_groups = []
        for g_i, g in enumerate(example_groups):
            enum_example_groups.append([g_i, g])
        group_combs = list(itertools.combinations(enum_example_groups, 1))
        for groups in group_combs:
            gcms = torch.cat([group[1]["gcm"] for group in groups], dim=0)
            max_o_num = max([len(g[1]["ocm"]) for g in groups])
            ocms = []
            for group in groups:
                if len(group[1]["ocm"]) < max_o_num:
                    ocms.append(
                        torch.cat((group[1]["ocm"], torch.zeros((max_o_num - len(group[1]["ocm"]), 10))), dim=0))
                else:
                    ocms.append(group[1]["ocm"])
            ocms = torch.stack(ocms)
            group_indices = [group[0] for group in groups]
            data = {"gcms": gcms, "ocms": ocms}
            group_clauses = df_search(args, lang, C, FC, data, group_num, level="group")
            gcs = {}
            for clause in group_clauses:
                # clause = clause_op.change_clause_obj_id(clause, args, group_indices, bk.variable_symbol_group)
                if clause not in gcs:
                    gcs[clause] = 1
                else:
                    gcs[clause] += 1
            group_data = {"group_clauses": gcs}
            example_clauses.append(group_data)
        all_clauses.append(example_clauses)
    removed_all_clauses = remove_trivial_clauses(all_clauses)
    lang.all_groups = removed_all_clauses
    # update language consts, atoms
    g_clauses = []

    for ic in removed_all_clauses:
        for g in ic:
            for g_clause, _ in g["group_clauses"].items():
                if g_clause not in g_clauses:
                    g_clauses.append(g_clause)
    lang.update_consts(g_clauses)
    lang.generate_atoms(g_clauses)
    lang.update_predicates(g_clauses)
    lang.clauses = g_clauses
    lang.save_lang(args, lang, mode, "group")

    return lang


def reasoning_rules(args, samples):
    """
    Derives common FOL clauses that are true in all positive samples but false in all negative samples,
    based on neuro and symbolic features.

    Args:
        samples (dict): A dictionary with two keys: "positive" and "negative".
            Each value is a dict with keys "neuro_features" and "symbolic_features".

    Returns:
        list: A list of FOL clause strings that represent the common logic rules.
    """
    # Extract features for positive and negative samples.
    pos_features = samples.get("positive", {})
    neg_features = samples.get("negative", {})
    # Step 1: Derive candidate FOL clauses from positive samples.
    candidate_clauses = derive_candidate_clauses(args, pos_features)

    # Step 2: Filter out candidate clauses that hold in negative samples.
    valid_clauses = []
    for clause in candidate_clauses:
        if not clause_holds(clause, neg_features):
            valid_clauses.append(clause)

    return valid_clauses


def derive_candidate_clauses(args, encoded_features):
    """
    Derives candidate FOL clauses based on positive sample features.

    This function analyzes the neuro and symbolic features from positive samples and
    generates a set of candidate FOL clauses that capture common patterns. The detailed
    implementation is omitted and can be filled in later.

    Args:
        encoded_features (dict): A dictionary with two keys: "symbolic" and "neuro".
            - symbolic: List of symbolic feature representations.
            - neuro: List of neuro feature representations (not used here).

    Returns:
        list: A list of candidate FOL clause strings.
    """

    symbolic_features = encoded_features.get("symbolic", [])[:3]
    neuro_features = encoded_features.get("neuro", [])[:3]  # For potential future use.

    # Initialize the language with necessary predicates.
    language = lang.Language()
    language.predicates = primitive.get_primitive_predicates()

    candidate_sets = []
    for scene_symbolic_features in symbolic_features:
        # Generate candidate clauses based on the symbolic features.
        candidate_clauses = language.generate_candidate_clauses(scene_symbolic_features)

        candidate_sets.append(candidate_clauses)

    common = filter_common_clauses_semantic_flexible(candidate_sets)
    # merged_clauses = language.merge_clauses(common)
    return common





def normalize_clause(clause):
    """
    Normalize a clause by removing variable indices.
    For example, "∃x1 in_pattern(x1)." becomes "∃x in_pattern(x)."
    """
    normalized = re.sub(r'x\d+', 'x', clause)
    return normalized


def parse_clause(clause):
    """
    Parses a clause string and returns a tuple (predicate, constant) if one exists.
    Expects a clause of the form:
       "<quantifiers>: predicate(x1, x2, ..., constant)"
    If no constant is present, returns (predicate, None).
    """
    # Remove quantifiers (assume quantifiers are before the colon).
    if ":" in clause:
        body = clause.split(":", 1)[1].strip()
    else:
        body = clause.strip()
    m = re.match(r'(\w+)\((.*)\)', body)
    if not m:
        return None
    predicate = m.group(1)
    args_str = m.group(2)
    args = [arg.strip() for arg in args_str.split(",")]
    if len(args) < 1:
        return (predicate, None)
    # If the last argument can be interpreted as a constant (e.g., not a variable like 'x'),
    # we assume it's the property constant.
    constant = args[-1]
    # If constant looks like a variable (e.g., "x"), then we treat it as None.
    if re.fullmatch(r'x', constant):
        constant = None
    return (predicate, constant)


def flatten_scene(scene):
    """
    Given a scene represented as a nested list of lists of clause strings (e.g., 4 lists, each with 11 clauses),
    flatten the structure and return a set of semantic pairs (predicate, constant) after normalization.
    """
    semantic_pairs = set()
    for sublist in scene:
        for clause in sublist:
            norm_clause = normalize_clause(clause)
            parsed = parse_clause(norm_clause)
            if parsed:
                semantic_pairs.add(parsed)
    return semantic_pairs


def filter_common_clauses_semantic_flexible(scenes):
    """
    Given multiple scenes (each represented as a nested list of lists of clause strings),
    this function identifies common semantic patterns without requiring the property constants to be identical.

    It builds, for each scene, a mapping from predicate name to the set of property constants that appear.
    Then for each predicate that exists in every scene, it checks if there is an intersection of constants.
      - If a common constant exists, the output clause will include it.
      - Otherwise, the output clause will be a generic one (without a constant).

    The output clause is generated in a fixed form with a minimal grouping, for example:
         "∃x1, x2: same_shape(x1, x2, circle)"  if a constant is common,
    or     "∃x1, x2: same_shape(x1, x2)"         if no common constant is found.

    Args:
        scenes (list): A list of scenes, where each scene is a nested list (e.g., 4 lists of 11 clauses).

    Returns:
        set: A set of new clause strings representing the common semantic patterns.
    """
    # Build mapping for each scene: predicate -> set of constants (None if no constant).
    scene_mappings = []
    for scene in scenes:
        semantic_pairs = flatten_scene(scene)  # set of (predicate, constant)
        mapping = {}
        for predicate, constant in semantic_pairs:
            mapping.setdefault(predicate, set()).add(constant)
        scene_mappings.append(mapping)

    # Determine which predicates are common to all scenes.
    common_predicates = set(scene_mappings[0].keys())
    for mapping in scene_mappings[1:]:
        common_predicates.intersection_update(mapping.keys())

    # For each common predicate, determine the common constant (if any).
    common_clauses = set()
    for pred in common_predicates:
        constant_sets = [mapping[pred] for mapping in scene_mappings]
        common_constants = set.intersection(*constant_sets)
        if common_constants and (None not in common_constants):
            # A common constant exists; choose one (arbitrarily).
            constant = list(common_constants)[0]
            clause = f"∃x1, x2: {pred}(x1, x2, {constant})"
        else:
            # No single common constant; output a generic clause.
            clause = f"∃x1, x2: {pred}(x1, x2)"
        common_clauses.add(clause)

    return common_clauses
