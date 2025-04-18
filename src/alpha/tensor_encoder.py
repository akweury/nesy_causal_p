# Created by X at 25.06.24

import itertools
import torch

from .fol import logic
from .fol import logic_ops


class TensorEncoder(object):
    """The tensor encoder for differentiable inference.

    A class for tensor encoding in differentiable forward-chaining approach.

    Args:
        lang (language): The language of first-order logic.
        facts (list(atom)): The set of ground atoms (facts).
        clauses (list(clause)): The set of clauses (rules).
        device (torch.device): The device to be used.

    Attrs:
        lang (language): The language of first-order logic.
        facts (list(atom)): The set of ground atoms (facts).
        clauses (list(clause)): The set of clauses (rules).
        G (int): The number of ground atoms.
        C (int): The number of clauses.
        L (int): The maximum length of the clauses.
        S (int): The maximum number of substitutions for body atoms.
        head_unifier_dic ({(atom, atom) -> List[(var, const)]}): The dictionary to save the computed unification results.
        fact_index_dic ({atom -> int}): The dictionary that maps an atom to its index.
    """

    def __init__(self, lang, facts, clauses, device):
        self.lang = lang
        self.facts = facts
        self.clauses = clauses
        self.device = device
        self.G = len(facts)
        self.C = len(clauses)
        # call before computing S and L
        # self.head_unifier_dic = self.build_head_unifier_dic()
        self.fact_index_dic = dict(zip(self.facts, list(range(len(self.facts)))))
        self.S = self.get_max_subs_num(clauses)
        self.L = max([len(clause.body) for clause in clauses] + [1])

    def get_max_subs_num(self, clauses):
        """Compute S (the maximum numebr of substitutions for body atoms) from clauses.

        Args:
            clauses (list(clause)): A set of clauses.

        Returns:
            S (int): The maximum number of substitutions for existentially quantified variables in the body atoms.
        """
        S_list = []
        for clause in clauses:
            # print("clause: ", clause)
            for fi, fact in enumerate(self.facts):
                # if (clause.head, fact) in self.head_unifier_dic:
                # theta = self.head_unifier_dic[(clause.head, fact)]
                unify_flag, theta = logic_ops.unify([clause.head, fact])
                if unify_flag:
                    clause_ = logic_ops.subs_list(clause, theta)
                    body = clause_.body
                    theta_list = self.generate_subs_multiple_dtype(body)
                    S_list.append(len(theta_list))
        return max(S_list)

    def encode(self):
        """Compute the index tensor for the differentiable inference.

        Returns
        I (tensor): The index tensor (G, C, S, L).
        """
        I = torch.zeros((self.C, self.G, self.S, self.L), dtype=torch.int16).to(self.device)
        for ci, clause in enumerate(self.clauses):
            # print("CLAUSE: ", clause)
            I_c = self.build_I_c(clause)
            I[ci, :, :, :] = I_c
        return I

    def build_I_c(self, clause):
        """Build index tensor for a given clause.

        Args:
            clause (clause): A clause.

        Returns:
            I_c (tensor): The index tensor for the given clause (G, S, L).
        """
        # G * S * L
        I_c = torch.zeros((self.G, self.S, self.L), dtype=torch.int16).to(self.device)
        # print("CLAUSE: ", clause)
        for fi, fact in enumerate(self.facts):
            # if (clause.head, fact) in self.head_unifier_dic:
            unify_flag, theta = logic_ops.unify([clause.head, fact])
            if unify_flag:
                # theta = self.head_unifier_dic[(clause.head, fact)]
                clause_ = logic_ops.subs_list(clause, theta)

                # I_c_b_file = str(config.buffer_path / "hide" / f"{args.dataset}" / f"I_c_b.pth.tar")
                # convert body atoms into indices
                buffer = self.body_to_tensor(clause_.body)
                I_c[fi] = buffer
        return I_c

    def build_fact_index_dic(self):
        """Build dictionary {fact -> index}

        Returns:
            dic ({atom -> int}): A dictionary to map the atoms to indices.
        """
        dic = dict(zip(self.facts, list(range(len(self.facts)))))
        return dic

    def build_head_unifier_dic(self):
        """Build dictionary {(head, fact) -> unifier}.

        Returns:
            dic ({(atom,atom) -> subtitution}): A dictionary to map the pair of ground atoms to their unifier.
        """
        dic = {}
        heads = set([c.head for c in self.clauses])
        for head in heads:
            for fi, fact in enumerate(self.facts):
                unify_flag, theta_list = logic_ops.unify([head, fact])
                if unify_flag:
                    dic[(head, fact)] = theta_list
        return dic

    # taking constant modes to reduce the number of substituions

    def body_to_tensor(self, body):
        """Convert the body atoms into a tensor.

        Args:
            body (list(atom)): The body atoms.

        Returns:
            I_c_b (tensor;(S * L)): The tensor representation of the body atoms.
        """
        # S * L

        I_c_b = torch.zeros((self.S, self.L), dtype=torch.int16).to(self.device)

        # extract all vars in the body atoms
        var_list = []
        for atom in body:
            var_list += atom.all_vars()
        var_list = list(set(var_list))

        assert len(var_list) <= 10, 'Too many existentially quantified variables in an atom: ' + str(atom)

        if len(var_list) == 0:
            # the case of the body atoms are already grounded
            x_b = self.facts_to_index(body)
            I_c_b[0] = self.pad_by_true(x_b)

            for i in range(1, self.S):
                I_c_b[i] = torch.zeros(self.L, dtype=torch.int16).to(self.device)  # fill by FALSE
        else:
            # the body has existentially quantified variable!!
            # e.g. body atoms: [in(img,O1),shape(O1,square)]
            # theta_list: [(O1,obj1), (O1,obj2)]
            theta_list = self.generate_subs_multiple_dtype(body)
            n_substs = len(theta_list)
            assert n_substs <= self.S, 'Exceeded the maximum number of substitution patterns to ' \
                                       'existential variables: n_substs is: ' \
                                       '' + str(n_substs) + ' but max num is: ' + str(self.S)

            # compute the grounded clause for each possible substitution, convert to the index tensor, and store it.
            for i, theta in enumerate(theta_list):
                ground_body = [logic_ops.subs_list(bi, theta) for bi in body]
                I_c_b[i] = self.pad_by_true(self.facts_to_index(ground_body))
            # if the number of substitutions is less than the maximum number of substitions (S),
            # the rest of the tensor is filled 0, which is the index of FALSE
            for i in range(n_substs, self.S):
                I_c_b[i] = torch.zeros(self.L, dtype=torch.int16).to(self.device)
        return I_c_b

    def body_to_tensor_faster(self, body, args):
        """Convert the body atoms into a tensor.

        Args:
            body (list(atom)): The body atoms.

        Returns:
            I_c_b (tensor;(S * L)): The tensor representation of the body atoms.
        """
        # S * L

        I_c_b = torch.zeros((self.S, self.L), dtype=torch.int16).to(self.device)

        # extract all vars in the body atoms
        var_list = []

        for atom in body:
            for t in atom.terms:
                if isinstance(t, logic.Var) and t not in var_list:
                    var_list.append(t)

        if len(var_list) == 0:
            # the case of the body atoms are already grounded
            x_b = self.facts_to_index(body)
            I_c_b[0] = self.pad_by_true(x_b)

            for i in range(1, self.S):
                I_c_b[i] = torch.zeros(self.L, dtype=torch.int16).to(self.device)  # fill by FALSE
        else:
            # the body has existentially quantified variable!!
            # e.g. body atoms: [in(img,O1),shape(O1,square)]
            # theta_list: [(O1,obj1), (O1,obj2)]
            theta_list = self.generate_subs_multiple_dtype(body)
            n_substs = len(theta_list)
            assert n_substs <= self.S, 'Exceeded the maximum number of substitution patterns to ' \
                                       'existential variables: n_substs is: ' \
                                       '' + str(n_substs) + ' but max num is: ' + str(self.S)

            # compute the grounded clause for each possible substitution, convert to the index tensor, and store it.
            for i, theta in enumerate(theta_list):
                ground_body = [logic_ops.subs_list(bi, theta) for bi in body]
                I_c_b[i] = self.pad_by_true(self.facts_to_index(ground_body))
            # if the number of substitutions is less than the maximum number of substitions (S),
            # the rest of the tensor is filled 0, which is the index of FALSE
            for i in range(n_substs, self.S):
                I_c_b[i] = torch.zeros(self.L, dtype=torch.int16).to(self.device)
        return I_c_b

    def pad_by_true(self, x):
        """Fill the tensor by ones for the clause which has less body atoms than the longest clause.

        Args:
            x (tensor): The tensor.

        Return:
            x_padded (tensor): The tensor that is padded to the shape of (S, L).
        """
        assert x.size(
            0) <= self.L, 'x.size(0) exceeds max_body_len: ' + str(self.L)
        if x.size(0) == self.L:
            return x
        else:
            diff = self.L - x.size(0)
            x_pad = torch.ones(diff, dtype=torch.int16).to(self.device)
            return torch.cat([x, x_pad])

    def get_var_list_from_body(self, body):
        var_dtype_list = []
        dtypes = []
        vars = []
        for atom in body:
            terms = atom.terms
            for i, term in enumerate(terms):
                if isinstance(term, tuple):
                    for t_i, t in enumerate(term):
                        if t.is_var():
                            dtype = atom.pred.sub_preds[i].dtypes[t_i]
                            var_dtype_list.append((t, dtype))
                            dtypes.append(dtype)
                            vars.append(t)

                elif term.is_var():
                    v = term
                    dtype = atom.pred.dtypes[i]
                    var_dtype_list.append((v, dtype))
                    dtypes.append(dtype)
                    vars.append(v)
        # in case there is no variables in the body
        if len(list(set(dtypes))) == 0:
            return []
        return vars, dtypes

    def get_consts_from_dtype(self, dtype):
        pass

    def generate_subs_multiple_dtype(self, body):
        # Define the variables and their domains
        variables, dtypes = self.get_var_list_from_body(body)
        # unique var and dtype
        var_unique = []
        var_unique_dtypes = []
        for i in range(len(variables)):
            if variables[i] not in var_unique:
                var_unique.append(variables[i])
                var_unique_dtypes.append(dtypes[i])
        # Create a list of domains
        const_domains = [self.lang.get_by_dtype(dtype) for dtype in var_unique_dtypes]
        # Generate the Cartesian product of the domains
        substitutions = list(itertools.product(*const_domains))

        def remove_same_elements_tuples(lst):
            return [tup for tup in lst if len(set(tup)) == len(tup)]

        substitutions = remove_same_elements_tuples(substitutions)

        theta_list = []
        # Store as tuples
        for substitution in substitutions:
            substitution_list = []
            for i in range(len(substitution)):
                substitution_list.append((var_unique[i], substitution[i]))
            if substitution_list not in theta_list:
                theta_list.append(substitution_list)
        return theta_list

    # taking constant modes to reduce the number of substitutions
    def generate_subs(self, body):
        """Generate substitutions from given body atoms.

        Generate the possible substitutions from given list of atoms. If the body contains any variables,
        then generate the substitutions by enumerating constants that matches the data type.
        !!! ASSUMPTION: The body has variables that have the same data type
            e.g. variables O1(object) and Y(color) cannot appear in one clause !!!

        Args:
            body (list(atom)): The body atoms which may contain existentially quantified variables.

        Returns:
            theta_list (list(substitution)): The list of substitutions of the given body atoms.
        """
        # extract all variables and corresponding data types from given body atoms
        var_dtype_list = []
        dtypes = []
        vars = []
        for atom in body:
            terms = atom.terms
            for i, term in enumerate(terms):
                if term.is_var():
                    v = term
                    dtype = atom.pred.dtypes[i]
                    var_dtype_list.append((v, dtype))
                    dtypes.append(dtype)
                    vars.append(v)
        # in case there is no variables in the body
        if len(list(set(dtypes))) == 0:
            return []
        # check the data type consistency
        # assert len(list(set(dtypes))) == 1, "Invalid existentially quantified variables. " + \
        #                                     str(len(list(set(dtypes)))) + " data types in the body: " + str(
        #     body) + " dypes: " + str(dtypes)

        vars = list(set(vars))
        n_vars = len(vars)
        consts = self.lang.get_by_dtype(dtypes[0])

        # e.g. if the data type is shape, then subs_consts_list = [(red,), (yellow,), (blue,)]
        subs_consts_list = list(itertools.permutations(consts, n_vars))

        theta_list = []
        # generate substitutions by combining variables to the head of subs_consts_list
        for subs_consts in subs_consts_list:
            theta = []
            for i, const in enumerate(subs_consts):
                s = (vars[i], const)
                theta.append(s)
            theta_list.append(theta)
        # e.g. theta_list: [[(Z, red)], [(Z, yellow)], [(Z, blue)]]
        # print("theta_list: ", theta_list)
        return theta_list

    def facts_to_index(self, atoms):
        """Convert given ground atoms into the indices.
        """

        ind = [self.get_fact_index(nf) for nf in atoms]
        if sum(ind)==0:
            raise ValueError
        return torch.tensor(ind, dtype=torch.int16).to(self.device)

    def get_fact_index_v(self, facts):
        """Convert a fact to the index in the ordered set of all facts.
        """
        try:
            index = self.fact_index_dic[facts]
        except KeyError:
            index = 0
        return index

    def get_fact_index(self, fact):
        """Convert a fact to the index in the ordered set of all facts.
        """
        try:
            index = self.fact_index_dic[fact]
        except KeyError:
            index = 0
        return index
