# Created by shaji at 24/06/2024
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
from skimage.transform import hough_line, hough_line_peaks

import config
from .fol import bk
from src.percept.perception import FCN
from src.utils import data_utils


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

    def __init__(self, pred_funs, lang, device):
        super().__init__()
        self.pred_funs = pred_funs
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device)
        # attr_term -> vector representation dic
        self.attrs = self.init_attr_encodings(device)


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


class VFLine(nn.Module):
    # check if og consists of lines and related to ig
    def __init__(self, name, device):
        super(VFLine, self).__init__()
        self.name = name
        self.device = device
        self.model = FCN().to(device)
        self.model.load_state_dict(torch.load(config.output / f'train_cha_line_groups' / 'line_detector_model.pth'))
        self.model.eval()  # Set the model to evaluation mode

    def find_lines(self, matrix):
        # Perform Hough Transform
        h, theta, d = hough_line(matrix.numpy())

        # Extract the angle and distance for the most prominent line
        accum, angles, dists = hough_line_peaks(h, theta, d)
        lines = [(angle, dist) for angle, dist in zip(angles, dists)]
        return lines

    def forward(self, group):
        # check lines
        g_patch = data_utils.group2patch(group["group_patch"], group["tile_pos"])
        g_tensor = data_utils.patch2tensor(g_patch)
        has_line = False
        line_conf = self.model(g_tensor.to(self.device).unsqueeze(0))
        if config.obj_true[line_conf.argmax()] == 1:
            has_line = True
            # find lines inside the patch
            lines = self.find_lines(g_tensor)
        return has_line


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


class VFHasFM(nn.Module):
    """The function v_in.
    """

    def __init__(self, name):
        super(VFHasFM, self).__init__()
        self.name = name

    def forward(self, fm, domain):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        prob = 1
        return prob


class VFRho(nn.Module):
    """The function v_area.
    """

    def __init__(self, name):
        super(VFRho, self).__init__()

        self.name = name

    def forward(self, z_1, z_2, dist_grade):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        dist_id = torch.zeros(rho.shape)

        dist_grade_num = dist_grade.shape[1]
        grade_weight = 1 / dist_grade_num
        for i in range(1, dist_grade_num):
            threshold = grade_weight * i
            dist_id[rho >= threshold] = i

        dist_pred = torch.zeros(dist_grade.shape).to(dist_grade.device)
        for i in range(dist_pred.shape[0]):
            dist_pred[i, int(dist_id[i])] = 1

        return (dist_grade * dist_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))



class VFPhi(nn.Module):
    """The function v_area.
    """

    def __init__(self, name):
        super(VFPhi, self).__init__()
        self.name = name

    def forward(self, z_1, z_2, dir):

        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        round_divide = dir.shape[1]
        area_angle = int(360 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi_clock_shift = (90 - phi.long()) % 360
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        dir_pred = torch.zeros(dir.shape).to(dir.device)
        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))



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


def get_valuation_module(args, lang):
    pred_funs = [
        VFHasFM('hasFM'),
        VFRho('rho'),
        VFPhi('phi')
    ]
    VM = FCNNValuationModule(pred_funs, lang=lang, device=args.device)
    return VM
