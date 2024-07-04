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

    def __init__(self, lang, device, dataset):
        super().__init__()
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

        v_color = FCNNColorValuationFunction()
        vfs['color_input'] = v_color
        layers.append(v_color)

        v_color = FCNNColorValuationFunction()
        vfs['color_output'] = v_color
        layers.append(v_color)

        v_duplicate = FCNNDuplicateValuationFunction()
        vfs['duplicate'] = v_duplicate
        layers.append(v_duplicate)

        v_scale = FCNNScaleOIValuationFunction()
        vfs['scale'] = v_scale
        layers.append(v_scale)

        v_shape = FCNNShapeValuationFunction()
        vfs['shape'] = v_shape
        layers.append(v_shape)

        v_hasIG = FCNNHasIGValuationFunction()
        vfs['hasIG'] = v_hasIG
        layers.append(v_hasIG)

        v_hasOG = FCNNHasOGValuationFunction()
        vfs['hasOG'] = v_hasOG
        layers.append(v_hasOG)

        # v_rho = FCNNRhoValuationFunction(device)
        # vfs['rho'] = v_rho
        # layers.append(v_rho)
        #
        # v_phi = FCNNPhiValuationFunction(device)
        # vfs['phi'] = v_phi
        # layers.append(v_phi)
        #
        # v_slope = FCNNSlopeValuationFunction(device)
        # vfs['slope'] = v_slope
        # layers.append(v_slope)

        v_shape_counter = FCNNShapeCounterValuationFunction()
        vfs['shape_counter'] = v_shape_counter
        layers.append(v_shape_counter)

        v_color_counter = FCNNColorCounterValuationFunction()
        vfs['color_counter'] = v_color_counter
        layers.append(v_color_counter)

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
            return torch.zeros((len(zs),)).to(torch.float32).to(self.device)

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
            try:
                group_data = [example_data["input_groups"][term_index] for example_data in data]
            except IndexError:
                raise IndexError
            return group_data
        elif term.dtype.name == "output_group":
            group_data = [example_data["output_groups"][term_index] for example_data in data]
            return group_data
        elif term.dtype.name in bk.attr_names:
            # return the standard attribute code
            return self.attrs[term].unsqueeze(0).repeat(len(data), 1)
        elif term.dtype.name == 'in_pattern':
            group_data = [example_data['input_groups'] for example_data in data]
            return group_data
        elif term.dtype.name == 'out_pattern':
            group_data = [example_data['output_groups'] for example_data in data]
            return group_data
        else:
            raise ValueError("Invalid datatype of the given term: " + str(term) + ':' + term.dtype.name)


class FCNNColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(FCNNColorValuationFunction, self).__init__()

    def forward(self, data, color_mask):
        """
        Args:
            data (tensor): 2-d tensor B * d of object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            color_mask (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        data_colors = torch.zeros_like(color_mask).to(color_mask.device)
        for d_i in range(len(data)):
            color = data[d_i]["color"]
            data_colors[d_i, color - 1] = 1
        is_color = (color_mask * data_colors).sum(dim=1)
        return is_color


class FCNNDuplicateValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNDuplicateValuationFunction, self).__init__()

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
        is_duplicate = torch.zeros(len(data_input))
        for e_i in range(len(data_input)):
            input_patch = data_input[e_i]["group_patch"]
            output_patch = data_output[e_i]["group_patch"]

            space_patch, target_patch = self.io2st_patch(input_patch, output_patch)
            duplicate_pos = self.find_identical_shape(space_patch, target_patch)
            if len(duplicate_pos) > 0:
                is_duplicate[e_i] = 1
        return is_duplicate


class FCNNScaleOIValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNScaleOIValuationFunction, self).__init__()

    def forward(self, data_input, data_output, scale_mask):

        data_scale = torch.zeros_like(scale_mask).to(scale_mask.device)
        for e_i in range(len(data_input)):
            input_patch = data_input[e_i]["group_patch"]
            output_patch = data_output[e_i]["group_patch"]
            if len(output_patch) % len(input_patch) == 0:
                data_scale[e_i, len(output_patch) // len(input_patch) - 1] = 1
            elif len(input_patch) % len(output_patch) == 0:
                data_scale[e_i, len(input_patch) // len(output_patch) - 1] = 1
        is_scale = (scale_mask * data_scale).sum(dim=1)
        return is_scale


class FCNNShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(FCNNShapeValuationFunction, self).__init__()

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


class FCNNShapeCounterValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNShapeCounterValuationFunction, self).__init__()

    def forward(self, z, a):
        attr_index = config.group_tensor_index["shape_counter"]
        z_shapeCounter = torch.zeros(a.shape)
        tensor_index = z[:, attr_index].to(torch.long)
        for i in range(len(tensor_index)):
            z_shapeCounter[i, tensor_index[i]] = 0.999
        return (a * z_shapeCounter).sum(dim=1)


class FCNNColorCounterValuationFunction(nn.Module):
    def __init__(self):
        super(FCNNColorCounterValuationFunction, self).__init__()

    def forward(self, z, a):
        attr_index = config.group_tensor_index["color_counter"]
        z_color_counter = torch.zeros(a.shape)
        tensor_index = z[:, attr_index].to(torch.long)
        for i in range(len(tensor_index)):
            z_color_counter[i, tensor_index[i]] = 0.999
        return (a * z_color_counter).sum(dim=1)


class FCNNHasIGValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(FCNNHasIGValuationFunction, self).__init__()

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
        prob = torch.zeros(len(ig))
        for e_i in range(len(ig)):
            if ig[e_i] in domain[e_i]:
                prob[e_i] = 1
        return prob


class FCNNHasOGValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(FCNNHasOGValuationFunction, self).__init__()

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
        prob = torch.zeros(len(og))
        for e_i in range(len(og)):
            if og[e_i] in domain[e_i]:
                prob[e_i] = 1
        return prob


class FCNNRhoValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNRhoValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

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


class FCNNPhiValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNPhiValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, dir):
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


class FCNNSlopeValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNSlopeValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, dir):
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
        dir_pred = torch.zeros(dir.shape).to(dir.device)

        z_without_line = z_1[:, config.group_tensor_index["line"]] == 0
        c_1 = self.get_left(z_1)
        c_2 = self.get_right(z_1)

        round_divide = dir.shape[1]
        area_angle = int(180 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi[phi < 0] = 360 - torch.abs(phi[phi < 0])
        phi_clock_shift = (90 + phi.long()) % 360
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        dir_pred[z_without_line] = 0

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def get_left(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_left_x"]], z[:, config.group_tensor_index["screen_left_y"]]))

    def get_right(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_right_x"]], z[:, config.group_tensor_index["screen_right_y"]]))


def get_valuation_module(args, lang, dataset):
    VM = FCNNValuationModule(lang=lang, device=args.device, dataset=dataset)
    return VM
