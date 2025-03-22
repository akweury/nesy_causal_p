# Created by X at 11.03.25


from torch import nn as nn
import torch
import torch.nn.functional as F

from src.beta.reasoning.lang import primitive_def
import src.beta.reasoning.lang.lang_bk as bk


class FCNNValuationModule(nn.Module):
    """
    A module to call valuation functions.

    Attributes:
        lang: The language.
        device: The device.
        layers (nn.ModuleList): List of valuation function modules.
        vfs (dict): Maps predicate names to their valuation function modules.
        attrs (dict): Maps attribute terms to their corresponding one-hot encoding or encoding info.
    """

    def __init__(self, pred_funs, device):
        super().__init__()
        self.pred_funs = pred_funs
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device)
        self.attrs = self.init_attr_encodings()

    def init_valuation_functions(self, device):
        """
        Initializes the valuation functions.

        Returns:
            layers (nn.ModuleList): List of valuation function modules.
            vfs (dict): Dictionary mapping predicate names to valuation functions.
        """

        vfs = {pred_fun.name: pred_fun for pred_fun in self.pred_funs}
        layers = nn.ModuleList(list(vfs.values()))
        return layers, vfs

    def init_attr_encodings(self):
        """
        Encodes attributes (e.g., color and shape) into one-hot encoding or encoding info.

        Returns:
            attrs (dict): Dictionary mapping an attribute term to its encoding.
        """
        attrs = {}
        for dtype_name in bk.attr_names:
            terms = self.lang.get_by_dtype_name(dtype_name)
            num_cls = len(terms)
            for term in terms:
                term_index = self.lang.term_index(term)
                attrs[term] = F.one_hot(torch.tensor(term_index), num_classes=num_cls).to(self.device)
        return attrs

    def forward(self, zs, atom):
        """
        Converts the object-centric representation to a valuation tensor.

        Args:
            zs (tensor): The object-centric representation (e.g., output from a YOLO model).
            atom: The target atom for which the probability is computed.

        Returns:
            tensor: A batch of probabilities for the target atom.
        """
        if atom.pred.name in self.vfs:
            pred_args = {}
            for term in atom.terms:
                term_name, term_data = self.ground_to_tensor(term, zs)
                pred_args[term_name] = term_data
            return self.vfs[atom.pred.name](pred_args)
        return torch.zeros(1, device=self.device)

    def ground_to_tensor(self, term, group_data):
        """
        Grounds a term into its tensor representation.

        Args:
            term: The term to be grounded.
            group_data (tensor): The object-centric representation.

        Returns:
            tuple: A tuple (term_name, term_data) representing the grounded term.
        """
        term_name = term.dtype.name
        if term_name in {"group_data", "group_objects_data", "pattern"}:
            return term_name, group_data
        elif term_name == "object":
            return term_name, self.lang.term_index(term)
        elif term_name in {
            bk.const_dtype_object_color,
            bk.const_dtype_object_shape,
            bk.const_dtype_group,
            bk.const_dtype_obj_num
        }:
            return term_name, term
        else:
            raise ValueError(f"Invalid datatype of the given term: {term} : {term.dtype.name}")


def get_primitive_predicates():
    pred_funs = [
        primitive_def.VFInP(),
        primitive_def.VFInG(),

        primitive_def.VFColor(),
        primitive_def.VFSameProperty(
            pred_name=bk.pred_names["same_color"],
            property_extractor=primitive_def.color_extractor,
            comparator=lambda c1, c2: primitive_def.color_comparator(c1, c2, threshold=5)
        ),

        primitive_def.VFShape(),
        primitive_def.VFSameProperty(
            pred_name=bk.pred_names["same_shape"],
            property_extractor=primitive_def.shape_extractor,
            comparator=primitive_def.shape_comparator
        ),

        primitive_def.VFSize(),
        primitive_def.VFSameProperty(
            pred_name=bk.pred_names["same_size"],
            property_extractor=primitive_def.size_extractor,
            comparator=lambda s1, s2: primitive_def.size_comparator(s1, s2, tolerance=0.1)
        ),

    ]

    return pred_funs
