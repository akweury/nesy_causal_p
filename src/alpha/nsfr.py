# Created by shaji at 24/06/2024

import numpy as np
import torch
from torch import nn as nn

from . import infer


# import aitk.utils.logic_utils as lu
def get_index_by_predname(pred_str, atoms):
    indices = []
    for p_i, p_str in enumerate(pred_str):
        p_indices = []
        for i, atom in enumerate(atoms):
            if atom.pred.name == p_str:
                p_indices.append(i)
        indices.append(p_indices)
    return indices


class NSFReasoner(nn.Module):
    """The Neuro-Symbolic Forward Reasoner.

    Args:
        perception_model (nn.Module): The perception model.
        facts_converter (nn.Module): The facts converter module.
        infer_module (nn.Module): The differentiable forward-chaining inference module.
        atoms (list(atom)): The set of ground atoms (facts).
    """

    def __init__(self, facts_converter, infer_module, clause_infer_module, atoms, clauses,
                 train=False):
        super().__init__()
        # self.pm = perception_module
        self.fc = facts_converter
        self.im = infer_module
        self.cim = clause_infer_module
        self.atoms = atoms
        self.bk = None
        self.clauses = clauses
        self._train = train

    def get_clauses(self):
        clause_ids = [np.argmax(w.detach().cpu().numpy()) for w in self.im.W]
        return [self.clauses[ci] for ci in clause_ids]

    def _summary(self):
        print("facts: ", len(self.atoms))
        print("I: ", self.im.I.shape)

    def get_params(self):
        return self.im.get_params()  # + self.fc.get_params()

    def forward(self, x):
        # obtain the object-centric representation
        # zs = self.pm(x)
        # convert to the valuation tensor

        # tic = time.perf_counter()

        V_0 = self.fc(x, self.atoms, self.bk)

        # toc = time.perf_counter()

        # a = V_0.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG

        # perform T-step forward-chaining reasoning
        V_T = self.im(V_0)

        # toc_2 = time.perf_counter()

        # print(f"Calculate V_0 in {toc - tic:0.4f} seconds")
        # print(f"Calculate V_T in {toc_2 - toc:0.4f} seconds")

        # tuple_list = []
        # for index in range(len(self.atoms)):
        #     tuple_list.append((self.atoms[index], V_T[0, index]))
        # b = V_T.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        return V_T

    # def clause_eval(self, x):
    #     # obtain the object-centric representation
    #     zs = self.pm(x)
    #     # convert to the valuation tensor
    #     V_0 = self.fc(zs, self.atoms, self.bk)
    #     # perform T-step forward-chaining reasoning
    #     V_T = self.cim(V_0)
    #     return V_T

    def clause_eval_quick(self, group):
        # x = torch.tensor(x)
        # convert to the valuation tensor
        V_0 = self.fc(group, self.atoms, self.bk)
        # perform T-step forward-chaining reasoning
        V_T = self.cim(V_0, self.atoms)
        return V_T

    def clause_eval_v_0(self, x):
        V_0 = self.fc(x, self.atoms, self.bk)
        return V_0

    def get_target_prediciton(self, atom_values, prednames, device):
        clause_num = atom_values.shape[0]
        example_num = atom_values.shape[1]
        atom_num = atom_values.shape[2]
        target_atom_num = len(prednames)
        target_prediction = torch.zeros(clause_num, example_num, target_atom_num).to(device)
        if target_atom_num > 1:
            target_indices = get_index_by_predname(pred_str=prednames, atoms=self.atoms)
            for t_i in range(len(target_indices)):
                target_prediction[:, :, t_i] = atom_values[:, :, target_indices[t_i]].max(dim=-1)[0]
                # target_prediction[0] = atom_values[0, :, target_indices[0]].max(dim=-1, keepdim=True)[0]
                # target_prediction[1:] = atom_values[1:, :, target_indices[1]].max(dim=-1, keepdim=True)[0]
        else:
            target_index_list = get_index_by_predname(pred_str=prednames, atoms=self.atoms)
            target_prediction = atom_values[:, :, target_index_list[0]]

        return target_prediction

    def get_test_target_prediciton(self, v, preds, device):
        """Extracting a value from the valuation tensor using a given predicate.
        """
        # v: batch * |atoms|
        values = torch.zeros(v.size(0), 1).to(device)

        target_indices = get_index_by_predname(pred_str=preds, atoms=self.atoms)
        # target_all = torch.zeros((len(target_indices), v.size(0)))
        # target_max = torch.zeros((len(target_indices)))

        # target_all[t_counter] = v[:, :, t_index]
        # target_max[t_counter] = v[:, :, t_index]
        # max_value = torch.max(target_max)
        result = v[target_indices[0]].max(dim=-1, keepdim=True)[0]

        return result

    def predict_multi(self, v, prednames):
        """Extracting values from the valuation tensor using given predicates.

        prednames = ['kp1', 'kp2', 'kp3']
        """
        # v: batch * |atoms|
        target_indices = []
        for predname in prednames:
            target_index = get_index_by_predname(
                pred_str=predname, atoms=self.atoms)
            target_indices.append(target_index)
        prob = torch.cat([v[:, i].unsqueeze(-1)
                          for i in target_indices], dim=1)
        B = v.size(0)
        N = len(prednames)
        assert prob.size(0) == B and prob.size(
            1) == N, 'Invalid shape in the prediction.'
        return prob

    def print_program(self):
        """Print asummary of logic programs using continuous weights.
        """
        print('====== LEARNED PROGRAM ======')
        C = self.clauses
        Ws_softmaxed = torch.softmax(self.im.W, 1)

        print("Ws_softmaxed: ", np.round(
            Ws_softmaxed.detach().cpu().numpy(), 2))

        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_' + str(i) + ': ',
                  C[max_i], np.round(np.array(W_[max_i].detach().cpu().item()), 2))

    def print_valuation_batch(self, valuation, n=40):
        # self.print_program()
        for b in range(valuation.size(0)):
            print('===== BATCH: ', b, '=====')
            v = valuation[b].detach().cpu().numpy()
            idxs = np.argsort(-v)
            for i in idxs:
                if v[i] > 0.1:
                    print(i, self.atoms[i], ': ', round(v[i], 3))

    def get_valuation_text(self, valuation):
        text_batch = ''  # texts for each batch
        for b in range(valuation.size(0)):
            top_atoms = self.get_top_atoms(valuation[b].detach().cpu().numpy())
            text = '----BATCH ' + str(b) + '----\n'
            text += self.atoms_to_text(top_atoms)
            text += '\n'
            # texts.append(text)
            text_batch += text
        return text_batch

    def get_top_atoms(self, v):
        top_atoms = []
        for i, atom in enumerate(self.atoms):
            if v[i] > 0.7:
                top_atoms.append(atom)
        return top_atoms

    def atoms_to_text(self, atoms):
        text = ''
        for atom in atoms:
            text += str(atom) + ', '
        return text


def get_nsfr_model(args, lang, FC, clauses, train=False):
    device = args.device
    atoms = lang.atoms

    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)

    IM = infer.build_infer_module(clauses, atoms, lang, m=len(prednames),
                                  infer_step=args.im_step, device=device, train=train, gamma=args.gamma)
    CIM = infer.build_clause_infer_module(args, clauses, atoms, lang, m=len(clauses),
                                          infer_step=args.cim_step, device=device, gamma=args.gamma)

    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, clause_infer_module=CIM,
                       atoms=atoms, clauses=clauses)
    return NSFR
