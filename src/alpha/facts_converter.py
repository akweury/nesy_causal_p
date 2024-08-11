# Created by shaji at 24/06/2024


import torch
import torch.nn as nn
from tqdm import tqdm

from .fol.logic import NeuralPredicate


class FactsConverter(nn.Module):
    """
    FactsConverter converts the output from the perception module to the valuation vector.
    """

    def __init__(self, args, lang, valuation_module):
        super(FactsConverter, self).__init__()
        # self.dim = args.d
        self.lang = lang
        self.vm = valuation_module  # valuation functions

        self.device = args.device

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.dim)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.dim)

    def forward(self, Z, raw_data, G, B, scores=None):
        return self.convert(Z, raw_data, G, B, scores)

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

    def convert(self, fms, raw_data, atoms, B, scores=None):
        example_num = raw_data.shape[0]
        # evaluate value of each atom
        V = torch.zeros((example_num, len(atoms))).to(torch.float32).to(self.device)
        # A is the covered area of each atom
        A = torch.zeros((example_num, len(atoms))).to(self.device)

        for i, atom in tqdm(enumerate(atoms), desc="Evaluating atoms..."):
            # this atom is a neural predicate
            if type(atom.pred) == NeuralPredicate and i > 1:
                try:
                    V[:, i] = self.vm(fms, raw_data, atom)
                except RuntimeError:
                    raise RuntimeError

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

        V[:, 1] = torch.ones((example_num,)).to(torch.float32).to(self.device)
        return V, A


def convert_i(self, zs, G):
    v = self.init_valuation(len(G))
    for i, atom in enumerate(G):
        if type(atom.pred) == NeuralPredicate and i > 1:
            v[i] = self.vm.eval(atom, zs)
    return v


def call(self, pred):
    return pred
