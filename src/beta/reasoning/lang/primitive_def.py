# Created by X at 11.03.25

import torch
from torch import nn as nn
import numpy as np
from src.beta.reasoning.lang import lang_bk as bk
from src.beta.beta_config import get_attribute_index


class BasePredicate:
    """
    Base class for all predicates. Automatically registers each predicate by name.
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically register the subclass if it defines 'pred_name'
        pred_name = getattr(cls, 'pred_name', None)
        if pred_name is not None:
            BasePredicate.registry[pred_name] = cls

    def __init__(self, name, arity):
        super(BasePredicate, self).__init__()
        self.name = name
        self.arity = arity
        # Register the predicate for easy retrieval
        BasePredicate.registry[name] = self.__class__

    def evaluate(self, ocm):
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def to_c(self, obj_props):
        raise NotImplementedError("Subclasses must implement the to_c method.")


class VFInP(BasePredicate):
    """
    Predicate: In-Pattern check.
    """

    def __init__(self):
        super(VFInP, self).__init__(bk.pred_names["in_pattern"], 1)

    def to_c(self, objs_existence):

        if len(list(set(objs_existence))) == 1:
            quantifiers = ["∀x"]
        else:
            quantifiers = [f"∃x" for i in range(len(objs_existence))]

        vars_str = [f"x" for i in range(len(objs_existence))]
        clauses = []
        for i in range(len(quantifiers)):
            var = vars_str[i]
            quantifier = quantifiers[i]
            if objs_existence[i]:
                clause = f"{quantifier} {self.name}({var})."
                clauses.append(clause)
        return clauses

    def evaluate(self, ocm):
        # Example implementation for in-pattern predicate.
        inp = ocm.sum(axis=-1) > 0
        cs = self.to_c(inp)
        return cs


class VFInG(BasePredicate):
    """
    Predicate: In-Group check.
    """

    def __init__(self):
        super(VFInG, self).__init__(bk.pred_names["in_group"], 1)

    def to_c(self, objs_existence):
        if len(list(set(objs_existence))) == 1:
            quantifiers = ["∀x"]
        else:
            quantifiers = [f"∃x" for i in range(len(objs_existence))]

        vars_str = [f"x" for i in range(len(objs_existence))]
        clauses = []
        for i in range(len(quantifiers)):
            var = vars_str[i]
            quantifier = quantifiers[i]
            if objs_existence[i]:
                clause = f"{quantifier} {self.name}({var})."
                clauses.append(clause)
        return clauses

    def evaluate(self, ocm):
        # Example implementation for in-group predicate.
        ing = ocm.sum(axis=-1) > 0
        cs = self.to_c(ing)
        return cs


class VFColor(BasePredicate):
    """
    Predicate: Checks if the object's color matches one of the expected colors.

    This predicate evaluates the object-centric matrix (20×10) using a flexible
    configuration. It determines the indices for the red, green, and blue channels
    from the configuration and then, for each non-padded object, finds the closest
    matching color in bk.color_matplotlib.
    """

    def __init__(self):
        # Initialize with the predicate name from bk.pred_names and arity = 1.
        super(VFColor, self).__init__(bk.pred_names["has_color"], 1)
        self.color_dict = bk.color_matplotlib  # For example: {"red": [255, 0, 0], "blue": [0, 0, 255], ...}
        # Determine color channel indices dynamically from the config.
        self.r_idx = get_attribute_index("color_r")
        self.g_idx = get_attribute_index("color_g")
        self.b_idx = get_attribute_index("color_b")

    def to_c(self, objs_color):

        if len(list(set(objs_color))) == 1:
            quantifiers = ["∀x"]
        else:
            quantifiers = [f"∃x" for i in range(len(objs_color))]

        vars_str = [f"x" for i in range(len(objs_color))]
        clauses = []
        for i in range(len(quantifiers)):
            var = vars_str[i]
            quantifier = quantifiers[i]
            color = objs_color[i]
            clause = f"{quantifier} {self.name}({var},{color})."
            clauses.append(clause)
        return clauses

    def evaluate(self, object_centric_matrix):
        """
        Given the object-centric matrix (20×10), determine the color of each object.
        Uses dynamic indices for the color channels as defined in the configuration.

        Args:
            object_centric_matrix (np.array): A 20×10 NumPy array where each row represents
                an object's encoded features.

        Returns:
            list: A list of unique color names that appear in the matrix.
        """
        obj_colors = []
        # Iterate over each row in the matrix.
        for row in object_centric_matrix:
            # Retrieve color channel values using indices from config.
            r = row[self.r_idx] if self.r_idx is not None else 0
            g = row[self.g_idx] if self.g_idx is not None else 0
            b = row[self.b_idx] if self.b_idx is not None else 0
            # Skip padded rows (assumed to be all zeros).
            if r == 0 and g == 0 and b == 0:
                continue
            color_vec = np.array([r, g, b])
            best_color = None
            best_distance = float('inf')
            # Compare the object's color with each target color.
            for color_name, target_rgb in self.color_dict.items():
                target_rgb = np.array(target_rgb)
                distance = np.linalg.norm(color_vec - target_rgb)
                if distance < best_distance:
                    best_distance = distance
                    best_color = color_name
            if best_color is not None:
                obj_colors.append(best_color)
        # Return unique colors that appear in the matrix.
        cs = self.to_c(obj_colors)
        return cs


class VFShape(BasePredicate):
    """
    Predicate: Checks the shape of objects.

    This predicate evaluates the object-centric matrix (20×10) using the shape indicator attributes.
    It retrieves the indices for "is_triangle", "is_square", and "is_circle" from the encoding configuration.
    For each non-padded object (i.e., each row that is not all zeros), it determines which shape flag is active
    and returns the unique set of shape names that appear in the matrix.
    """

    def __init__(self):
        # Initialize with the predicate name from bk.pred_names and arity = 1.
        super(VFShape, self).__init__(bk.pred_names["has_shape"], 1)
        # Retrieve indices dynamically from the configuration.
        self.triangle_idx = get_attribute_index("is_triangle")
        self.square_idx = get_attribute_index("is_square")
        self.circle_idx = get_attribute_index("is_circle")

    def to_c(self, objs_shape):
        if len(list(set(objs_shape))) == 1:
            quantifiers = ["∀x"]
        else:
            quantifiers = [f"∃x" for i in range(len(objs_shape))]

        vars_str = [f"x" for i in range(len(objs_shape))]
        clauses = []
        for i in range(len(quantifiers)):
            var = vars_str[i]
            quantifier = quantifiers[i]
            shape = objs_shape[i]
            clause = f"{quantifier} {self.name}({var},{shape})."
            clauses.append(clause)
        return clauses

    def evaluate(self, object_centric_matrix):
        """
        Given the object-centric matrix (20×10), determine the shape of each object.

        Args:
            object_centric_matrix (np.array): A 20×10 NumPy array where each row represents
                an object's encoded features.

        Returns:
            list: A list of unique shape names (e.g. "triangle", "square", "circle") that appear in the matrix.
        """
        shapes_found = []
        # Iterate over each object (row) in the matrix.
        for row in object_centric_matrix:
            # Skip padded rows (assumed to be all zeros).
            if np.all(row == 0):
                continue

            # Check each shape indicator if its corresponding column exists.
            if self.triangle_idx is not None and row[self.triangle_idx] == 1:
                shapes_found.append("triangle")
            elif self.square_idx is not None and row[self.square_idx] == 1:
                shapes_found.append("square")
            elif self.circle_idx is not None and row[self.circle_idx] == 1:
                shapes_found.append("circle")
            # If no flag is set, we may skip or consider it as "undefined".

        # Return unique shapes found in the matrix.
        cs = self.to_c(shapes_found)
        return cs


class VFSize(BasePredicate):
    """
    Predicate: Checks if the object's size falls within the common (similar) range.

    This predicate takes a 20×10 object-centric matrix as input, where the "size" attribute
    is stored in column index 2. It analyzes the sizes of all valid (non-padded) objects.
    If most objects have similar sizes (i.e. the interquartile range is small relative to the median),
    then the similar size range is defined as [Q1, Q3]. The predicate returns a list of boolean values,
    one for each row of the matrix, with True indicating the object's size lies within the similar range.

    If the sizes are too variable (i.e. no clear similar range exists), then all truth values are set to False.
    """

    def __init__(self):
        super(VFSize, self).__init__(bk.pred_names["has_size"], 1)

    def to_c(self, obj_sizes):
        if len(list(set(obj_sizes))) == 1:
            quantifiers = ["∀x"]
        else:
            quantifiers = [f"∃x" for i in range(len(obj_sizes))]

        vars_str = [f"x" for i in range(len(obj_sizes))]
        clauses = []
        for i in range(len(quantifiers)):
            var = vars_str[i]
            quantifier = quantifiers[i]
            obj_size = obj_sizes[i]
            clause = f"{quantifier} {self.name}({var},{obj_size})."
            clauses.append(clause)
        return clauses

    def evaluate(self, object_centric_matrix):
        """
        Evaluate the sizes of objects in the given object-centric matrix.

        Args:
            object_centric_matrix (np.array): A 20×10 matrix of object features.
                Assumes that the size is stored at column index 2.

        Returns:
            list: A list of boolean values (length equal to the number of rows in the matrix)
                  indicating whether each object's size falls within the detected similar range.
        """
        sizes = []
        valid_indices = []
        # Collect sizes for valid objects (assume padded objects have size == 0)
        for i, row in enumerate(object_centric_matrix):
            size = row[2]
            if size > 0:
                sizes.append(size)
                valid_indices.append(i)

        # If no valid objects, return all False.
        if len(sizes) == 0:
            return [False] * object_centric_matrix.shape[0]

        sizes = np.array(sizes)
        Q1 = np.percentile(sizes, 25)
        Q3 = np.percentile(sizes, 75)
        median = np.median(sizes)
        iqr = Q3 - Q1

        # Define a threshold ratio (e.g., 20% of median) to decide if most objects have similar size.
        similarity_ratio_threshold = 0.2
        if median == 0 or (iqr / median) >= similarity_ratio_threshold:
            # Sizes are too variable (or median is zero), so no common size range.
            return [False] * object_centric_matrix.shape[0]

        # Otherwise, define the similar range as [Q1, Q3]
        similar_range = (Q1, Q3)

        # For each row, check if the object's size is within the similar range.
        obj_sizes = []
        for row in object_centric_matrix:
            if np.all(row == 0):
                continue
            size = row[2]
            if size == 0:
                obj_sizes.append(False)
            else:
                if similar_range[0] <= size <= similar_range[1]:
                    obj_sizes.append(similar_range)
                else:
                    obj_sizes.append(False)
        clauses = self.to_c(obj_sizes)
        return clauses


class VFSameProperty(BasePredicate):
    """
    A generic predicate for determining if a group of objects share the same property.

    This predicate is parameterized by:
      - property_extractor: a function that extracts the property from an object's feature vector.
      - comparator: a function that takes two property values and returns True if they are considered similar.

    The evaluate method groups objects from the object-centric matrix (20x10) that share similar property
    values and returns a set of clauses. Each clause is of the form:

        "∃x1, x2, ..., xN: <predicate_name>(x1, x2, ..., xN, <property_constant>)"

    where <property_constant> is a string representing the shared property (e.g. "triangle" or "5.0").
    """

    def __init__(self, pred_name, property_extractor, comparator, arity=1):
        super(VFSameProperty, self).__init__(pred_name, arity)
        self.property_extractor = property_extractor
        self.comparator = comparator

    def evaluate(self, object_centric_matrix):
        """
        Group objects with similar property values and generate a clause for each group that has at least 2 objects.

        Args:
            object_centric_matrix (np.array): A 20x10 array of encoded object features.

        Returns:
            set: A set of clause strings, each declaring that a group of objects share the same property.
                 For example:
                     "∃x1, x2, x3: same_shape(x1, x2, x3, triangle)"
        """
        num_objects = object_centric_matrix.shape[0]
        properties = [None] * num_objects
        valid_indices = []
        # Extract the property for each object.
        for i in range(num_objects):
            prop = self.property_extractor(object_centric_matrix[i])
            # Consider an object valid if its property is not None and,
            # for numeric properties, not zero.
            if prop is None or (isinstance(prop, (int, float)) and prop == 0):
                properties[i] = None
            else:
                properties[i] = prop
                valid_indices.append(i)

        # Group valid objects by similarity.
        # Each group will be stored as a tuple: (list of indices, representative property)
        groups = []
        used = set()
        for idx in valid_indices:
            if idx in used:
                continue
            group = [idx]
            used.add(idx)
            rep_prop = properties[idx]  # Use the first object's property as the representative.
            for j in valid_indices:
                if j in used:
                    continue
                if self.comparator(rep_prop, properties[j]):
                    group.append(j)
                    used.add(j)
            groups.append((group, rep_prop))

        # Generate clauses for each group with at least 2 objects.
        clauses = set()
        for group, prop in groups:
            if len(group) < 2:
                continue
            # Create distinct variable names for the objects in this group.
            var_list = [f"x{i + 1}" for i in range(len(group))]
            # Convert the property value to a string constant.
            prop_str = str(prop)
            # Build the clause string.
            clause = f"∃{', '.join(var_list)}: {self.name}({', '.join(var_list)}, {prop_str})"
            clauses.add(clause)
        return clauses


# --- Example specialized comparators and property extractors ---

# For size (numeric), we compare with a relative tolerance.
def size_extractor(row):
    # Assuming "size" is in column index 2.
    return row[2]


def size_comparator(s1, s2, tolerance=0.1):
    # Avoid division by zero.
    if s1 == 0:
        return False
    return abs(s1 - s2) / s1 <= tolerance


# For shape (categorical), we simply check for equality.
def shape_extractor(row):
    # Assume shape is encoded as one-hot in three columns.
    # We return a string based on which column has 1.
    # For example, assume indices 7, 8, 9 correspond to triangle, square, circle.
    if row[7] == 1:
        return "triangle"
    elif row[8] == 1:
        return "square"
    elif row[9] == 1:
        return "circle"
    else:
        return None


def shape_comparator(s1, s2):
    return s1 == s2


# For color, assume color is represented as an RGB tuple from columns 4, 5, 6.
def color_extractor(row):
    r, g, b = row[4], row[5], row[6]
    # If all channels are zero, consider it invalid.
    if r == 0 and g == 0 and b == 0:
        return None
    return (r, g, b)


def color_comparator(c1, c2, threshold=50):
    c1 = np.array(c1)
    c2 = np.array(c2)
    distance = np.linalg.norm(c1 - c2)
    return distance < threshold


# --- Creating specialized instances ---

# Create a predicate for "same size"
VFSameSize = VFSameProperty(
    pred_name=bk.pred_names["same_size"],
    property_extractor=size_extractor,
    comparator=lambda s1, s2: size_comparator(s1, s2, tolerance=0.1)
)

# Create a predicate for "same shape"
VFSameShape = VFSameProperty(
    pred_name=bk.pred_names["same_shape"],
    property_extractor=shape_extractor,
    comparator=shape_comparator
)

# Create a predicate for "same color"
VFSameColor = VFSameProperty(
    pred_name=bk.pred_names["same_color"],
    property_extractor=color_extractor,
    comparator=lambda c1, c2: color_comparator(c1, c2, threshold=50)
)

