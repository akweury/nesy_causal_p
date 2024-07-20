# Created by shaji at 24/06/2024
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config
from .fol import bk


class FCNNValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, pred_funs, lang, device, dataset):
        super().__init__()
        self.pred_funs = pred_funs
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device)
        # attr_term -> vector representation dic
        self.attrs = self.init_attr_encodings(device)
        self.dataset = dataset

    def init_valuation_functions(self, device):
        """
            Args:
                device (device): The device.
                dataset (str): The dataset.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # a dictionary: pred_name -> valuation function
        for pred_fun in self.pred_funs:
            vfs[pred_fun.name] = pred_fun
            layers.append(vfs[pred_fun.name])
        return nn.ModuleList(layers), vfs

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
                term_index = torch.tensor(self.lang.term_index(term)).to(device)
                num_cls = len(self.lang.get_by_dtype_name(dtype_name))
                attrs[term] = F.one_hot(term_index, num_classes=num_cls).to(device)
        return attrs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        if atom.pred.name in self.vfs:
            args = [self.ground_to_tensor(term, zs) for term in atom.terms]
            # call valuation function
            return self.vfs[atom.pred.name](*args)
        else:
            return torch.zeros((1,)).to(torch.float32).to(self.device)

    def ground_to_tensor(self, term, data):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                data (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'input_group':
            # return the coding of the group
            group_data = data["input_groups"][term_index]
            return group_data
        elif term.dtype.name == "output_group":
            group_data = data["output_groups"][term_index]
            return group_data
        elif term.dtype.name in bk.attr_names:
            # return the standard attribute code
            return self.attrs[term].unsqueeze(0)
        elif term.dtype.name == 'in_pattern':
            group_data = data['input_groups']
            return group_data
        elif term.dtype.name == 'out_pattern':
            group_data = data['output_groups']
            return group_data
        else:
            raise ValueError("Invalid datatype of the given term: " + str(term) + ':' + term.dtype.name)


class VFColor(nn.Module):
    """The function v_color.
    """

    def __init__(self, name):
        super(VFColor, self).__init__()
        self.name = name

    def forward(self, data, color_mask):
        data_colors = torch.zeros_like(color_mask).to(color_mask.device)

        color = data["color"]
        data_colors[0, color - 1] = 1
        is_color = (color_mask * data_colors).sum(dim=1)
        return is_color


class VFDuplicate(nn.Module):
    def __init__(self, name):
        super(VFDuplicate, self).__init__()
        self.name = name

    def io2st_patch(self, input_patch, output_patch):
        if input_patch.shape == output_patch.shape:
            # find identical patch in state B
            space_patch = output_patch
            target_patch = input_patch
        elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
            # input shape is bigger than output
            space_patch = input_patch
            target_patch = output_patch
        else:
            space_patch = output_patch
            target_patch = input_patch
        return space_patch, target_patch

    def find_identical_shape(self, space_patch, target_patch):
        """ the small_array can be x% identical to large_array
        return: top n identical patches in large array, its position, width, and different tiles if any
        """
        space_patch = np.array(space_patch)
        target_patch = np.array(target_patch)
        # map all number to 1 (bw mode)
        space_patch = space_patch / (space_patch + 1e-20)
        target_patch = target_patch / (target_patch + 1e-20)
        # Get sliding windows of shape (3, 3) from the large array
        windows = sliding_window_view(space_patch, target_patch.shape)
        # Calculate the similarity percentage for each window
        match_counts = np.sum(windows == target_patch, axis=(2, 3))
        similarity = match_counts / target_patch.size * 100
        # Generate the positions and differences
        positions = np.argwhere(similarity == 100)
        return positions

    def forward(self, data_input, data_output):
        is_duplicate = 0

        input_patch = data_input["group_patch"]
        output_patch = data_output["group_patch"]

        space_patch, target_patch = self.io2st_patch(input_patch, output_patch)
        duplicate_pos = self.find_identical_shape(space_patch, target_patch)
        if len(duplicate_pos) > 0:
            is_duplicate = 1
        return is_duplicate


class VFScale(nn.Module):
    def __init__(self, name):
        super(VFScale, self).__init__()
        self.name = name

    def forward(self, data_input, data_output, scale_mask):
        data_scale = torch.zeros_like(scale_mask).to(scale_mask.device)

        input_patch = data_input["group_patch"]
        output_patch = data_output["group_patch"]
        if len(output_patch) % len(input_patch) == 0:
            data_scale[0, len(output_patch) // len(input_patch) - 1] = 1
        elif len(input_patch) % len(output_patch) == 0:
            data_scale[0, len(input_patch) // len(output_patch) - 1] = 1
        is_scale = (scale_mask * data_scale).sum(dim=1)
        return is_scale[0]


class VFFulfil(nn.Module):
    """ group_a and group_b are monotonous color groups.
    Return true if group_b fulfills group_a """

    def __init__(self, name):
        super(VFFulfil, self).__init__()
        self.name = name

    def flood_fill(self, matrix, start):
        n = len(matrix)
        m = len(matrix[0])
        if n == 0 or m == 0:
            return []

        x, y = start
        target_value = matrix[x][y]
        visited = [[False for _ in range(m)] for _ in range(n)]
        connected_items = []

        def dfs(x, y):
            if x < 0 or x >= n or y < 0 or y >= m:
                return
            if visited[x][y] or matrix[x][y] != target_value:
                return
            visited[x][y] = True
            connected_items.append((x, y))
            # Explore 4-connected neighbors
            for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                dfs(x + dx, y + dy)

        dfs(x, y)
        return connected_items

    def forward(self, group_a, group_b):
        # find all the holes of group a say, gorup_c
        flood_tiles = []
        group_b_positions = group_b["tile_pos"]
        for pos in group_b_positions:
            if pos not in flood_tiles:
                flood = self.flood_fill(group_a["group_patch"], pos)
                flood_tiles += flood
        if flood_tiles == group_b_positions:
            return True
        else:
            return False


class VFSurround(nn.Module):
    def __init__(self, name):
        super(VFSurround, self).__init__()
        self.name = name

    def find_boundaries(self, matrix, group):
        boundaries = []
        n, m = matrix.shape

        def find_tile_boundary(pos, group):
            x, y = pos
            boundary = []
            for dx, dy in bk.neighbor_4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in group:
                    boundary.append((nx, ny))
            return boundary

        for pos in group:
            tile_boundary = find_tile_boundary(pos, group)
            boundaries += [b for b in tile_boundary if b not in boundaries]
        return boundaries

    def forward(self, group_a, group_b):
        # if group_a is the boundary of group_b
        boundary_b = self.find_boundaries(group_b["group_patch"], group_b["tile_pos"])

        if boundary_b == group_a["tile_pos"]:

            return True
        else:
            return False


class FCNNShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self, name):
        super(FCNNShapeValuationFunction, self).__init__()
        self.name = name

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        shape_indices = [config.group_tensor_index[_shape] for _shape in config.group_pred_shapes]
        z_shape = z[:, shape_indices]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class VFHasIG(nn.Module):
    """The function v_in.
    """

    def __init__(self, name):
        super(VFHasIG, self).__init__()
        self.name = name

    def forward(self, ig, domain):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        prob = 0
        if ig in domain:
            prob = 1
        return prob


class VFHasOG(nn.Module):
    """The function v_in.
    """

    def __init__(self, name):
        super(VFHasOG, self).__init__()
        self.name = name

    def forward(self, og, domain):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        prob = 0
        if og in domain:
            prob = 1
        return prob


class VFRepeat(nn.Module):
    def __init__(self, name):
        super(VFRepeat, self).__init__()
        self.name = name

    def io2st_patch(self, input_patch, output_patch):
        if input_patch.shape == output_patch.shape:
            # find identical patch in state B
            space_patch = output_patch
            target_patch = input_patch
        elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
            # input shape is bigger than output
            space_patch = input_patch
            target_patch = output_patch
        else:
            space_patch = output_patch
            target_patch = input_patch
        return space_patch, target_patch

    def has_repeating_patterns(self, matrix):
        n = len(matrix)

        # Function to extract a kxk submatrix as a tuple
        def extract_submatrix(matrix, start_row, start_col, k):
            return tuple(
                tuple(matrix[i][j] for j in range(start_col, start_col + k)) for i in range(start_row, start_row + k))

        # Iterate over all possible submatrix sizes
        for k in range(1, n + 1):
            seen_submatrices = set()

            # Slide the kxk window over the matrix
            for i in range(n - k + 1):
                for j in range(n - k + 1):
                    submatrix = extract_submatrix(matrix, i, j, k)
                    if submatrix in seen_submatrices:
                        return True
                    seen_submatrices.add(submatrix)

        return False

    def forward(self, data_input, data_output):
        is_repeat = 0
        input_patch = data_input["group_patch"]
        output_patch = data_output["group_patch"]
        space_patch, target_patch = self.io2st_patch(input_patch, output_patch)

        # check if they are repeat patterns
        has_rp_input = self.has_repeating_patterns(space_patch)
        has_rp_output = self.has_repeating_patterns(target_patch)

        if has_rp_input and has_rp_output:
            is_repeat = 1
        return is_repeat


def get_valuation_module(args, lang, dataset):
    pred_funs = [
        VFHasIG('hasIG'),
        VFHasOG('hasOG'),
        VFScale('scale'),
        VFColor('color'),
        VFFulfil('fulfil'),
        VFDuplicate('duplicate'),
        VFRepeat('repeat'),
        VFSurround('surround'),
    ]
    VM = FCNNValuationModule(pred_funs, lang=lang, device=args.device, dataset=dataset)
    return VM
