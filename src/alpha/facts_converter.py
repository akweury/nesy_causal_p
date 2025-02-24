# Created by X at 24/06/2024

import torch
import torch.nn as nn
from tqdm import tqdm

from .fol.logic import NeuralPredicate, InventedPredicate
from src import bk
from src.alpha import valuation
from src.alpha.fol.logic import Var

class FactsConverter(nn.Module):
    """
    FactsConverter converts the output from the perception module to the valuation vector.
    """

    def __init__(self, args, lang, valuation_module, given_attrs=None):
        super(FactsConverter, self).__init__()
        # self.dim = args.d
        self.args = args
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = args.device
        if given_attrs is None:
            self.attrs = self.init_attr_encodings(args.device)
            lang.attrs = self.attrs
        else:
            self.attrs = given_attrs

    def obj_mode(self):
        self.vm.vfs = obj_vfs

    def init_attr_encodings(self, device):
        """Encode color and shape into one-hot encoding.

            Args:
                device (device): The device.

            Returns:
                attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
        """
        attr_names = bk.attr_names
        attrs = {}
        for dtype_name in attr_names:
            for term in self.lang.get_by_dtype_name(dtype_name):
                term_index = torch.tensor(self.lang.term_index(term)).tolist()
                num_cls = len(self.lang.get_by_dtype_name(dtype_name))
                # attrs[term] = F.one_hot(term_index, num_classes=num_cls).to(device)
                attrs[term] = [term_index, num_cls]
        return attrs

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.dim)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.dim)

    def forward(self, Z, G, B, scores=None):
        return self.convert(Z, G, B, scores)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def ground_to_tensor(self, term, group_data):
        if isinstance(term, Var):
            term_name = term.var_type
        else:
            term_name = term.dtype.name
        term_data = None
        if term_name == "group_data":
            # self.group_indices = group_data[:, bk.prop_idx_dict["group_name"]] > 0
            term_data = group_data
            term_name = "group_data"
        elif term_name == "object":
            term_data = group_data
        elif term_name in [bk.const_dtype_object_color, bk.const_dtype_object_shape, bk.const_dtype_group,
                           bk.const_dtype_obj_num]:
            term_data = term
        # elif term_name in bk.attr_names:
        # return the standard attribute code
        # term_data = self.attrs[term]
        elif term_name == 'pattern':
            # return the image
            term_data = group_data
        else:
            raise ValueError("Invalid datatype of the given term: " + str(term) + ':' + term.dtype.name)
        return term_name, term_data

    def convert(self, group, atoms, B, scores=None):
        # evaluate value of each atom
        V = torch.zeros((1, len(atoms))).to(torch.float32).to(self.device)
        for i, atom in enumerate(atoms):
            # this atom is a neural predicate
            if type(atom.pred) == NeuralPredicate and i > 1:
                V[:, i] = self.vm(group, atom)
            elif type(atom.pred) == InventedPredicate:

                # collect the term data of the atom
                pred_args = {}
                terms = atom.terms
                if not isinstance(terms, tuple):
                    raise ValueError
                for t in terms:
                    term_name, term_data = self.ground_to_tensor(t, group)
                    pred_args[term_name] = term_data

                # collecting the data
                atom_conf = 1.0
                for a_i in range(len(atom.pred.sub_preds)):
                    # if isinstance(atom.terms[a_i], Const):
                    #     term = atom.terms
                    # elif isinstance(atom.terms[a_i], list):
                    #     term = atom.terms[a_i]
                    # elif isinstance(atom.terms[a_i], tuple):
                    #     term = atom.terms[a_i]
                    # else:
                    #     raise ValueError
                    # for t in term:
                    #     term_name, term_data = self.ground_to_tensor(t, group)
                    #     pred_args[term_name] = term_data
                    module_name = atom.pred.sub_preds[a_i].name
                    # valuating via the predicate mechanics

                    module = valuation.valuation_modules[module_name]
                    module_res = module(pred_args)
                    try:
                        atom_conf *= module_res
                    except RuntimeError:
                        raise RuntimeError
                # self.args.logger.debug(f"(atom) {atom_conf:.1f} {atom} ")
                V[:, i] = atom_conf

            # this atom is an invented predicate
            # elif type(atom.pred) == InventedPredicate:
            #     if atom.pred.body is not None:
            #         value = self.pi_vm(atom, atom.pred.body, V, G)
            #         V[:, i] = value

            # this atom in background knowledge
            # elif atom in B:
            # # V[:, i] += 1.0
            #     value = torch.ones((batch_size,)).to(torch.float32).to(self.device)
            #     V[:, i] += value

        V[:, 1] = torch.ones((1,)).to(torch.float32).to(self.device)
        return V


def convert_i(self, zs, G):
    v = self.init_valuation(len(G))
    for i, atom in enumerate(G):
        if type(atom.pred) == NeuralPredicate and i > 1:
            v[i] = self.vm.eval(atom, zs)
    return v


def call(self, pred):
    return pred
