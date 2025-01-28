from PIL import Image, ImageDraw
import numpy as np
import os
import json
from pathlib import Path
import torch
from fontTools.svgLib.path import shapes

import config
from tqdm import tqdm
from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.ShapeOnShapes import ShapeOnShape
from src import bk
from src.utils import chart_utils, args_utils
from src.percept import gestalt_group
from kandinsky_generator import gestalt_patterns

u = KandinskyUniverse.SimpleUniverse()


###
# Parameters for images: object sizes, object colors, background colors
###
# WIDTH = 640
#
# MINSIZE = 10 * 5
# MAXSIZE = 24 * 5

# clevr
# kandinsky_colors = {"red": (173, 35, 35),
#                     "yellow": (255, 238, 51), "blue": (42, 75, 215)}

# a bit lighter
# kandinsky_colors = [(193, 85, 85), (255, 238, 101), (90, 135, 235)]
# background = (215, 215, 215, 255)


# def overlaps(objs):
#     image = Image.new("L", (WIDTH, WIDTH), 0)
#     sumarray = np.array(image)
#     d = ImageDraw.Draw(image)
#
#     for obj in objs:
#         image = Image.new("L", (WIDTH, WIDTH), 0)
#         draw = ImageDraw.Draw(image)
#         utils.draw_obj(draw, obj['shape'], obj['size'], 10, obj["cx"], obj["cy"])
#
#         sumarray += np.array(image)
#
#     sumimage = Image.fromarray(sumarray)
#     return sumimage.getextrema()[1] > 10


# def kandinskyFigure(objs, color_dict, subsampling=1):
#     image = Image.new("RGB", (subsampling * WIDTH, subsampling * WIDTH), background)
#     draw = ImageDraw.Draw(image)
#     for obj in objs:
#         obj_size = subsampling * obj['size']
#         obj_color = color_dict[obj["color"]]
#         obj_x = subsampling * obj['cx']
#         obj_y = subsampling * obj['cy']
#         utils.draw_obj(draw, obj['shape'], obj_size, obj_color, obj_x, obj_y)
#     if subsampling > 1:
#         image = image.resize((WIDTH, WIDTH), Image.BICUBIC)
#     return image


# def generate_imgs(dataset_name, all_data, save_path, color_dict):
#     for d_i, data in enumerate(all_data):
#         image = kandinskyFigure(data, color_dict, 1)
#
#         img_data = kandinskyData(data, color_dict, image.width)
#
#         # save image, masks and labels
#         image.save(save_path / f"{dataset_name}_{d_i:03d}.png")
#         with open(save_path / f"{dataset_name}_{d_i:03d}.json", 'w') as f:
#             json.dump(img_data, f)
#         print(f"Successfully saved image {d_i}.")


# def to_images(kfgen, n=50, width=640):
#     pos_imgs = []
#     neg_imgs = []
#     for (i, kf) in enumerate(kfgen.true_kf(n)):
#         image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
#         pos_imgs.append(image)
#
#     for (i, kf) in enumerate(kfgen.false_kf(n)):
#         image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
#         neg_imgs.append(image)
#     return pos_imgs, neg_imgs


# def kandinskyData(objs, color_dict, width):
#     data = []
#     for obj in objs:
#         data.append({"x": obj["cx"],
#                      "y": obj["cy"],
#                      "size": obj['size'],
#                      "color_name": obj["color"],
#                      "color_r": color_dict[obj["color"]][0],
#                      "color_g": color_dict[obj["color"]][1],
#                      "color_b": color_dict[obj["color"]][2],
#                      "shape": obj["shape"],
#                      "width": width
#                      })
#     return data


# def main(dataset, min_obj_num, max_obj_num, color_dict, mode='train', img_num=1):
#     # save iamges into true/false folders
#     base_path = config.kp_dataset / dataset
#     true_path = base_path / mode / 'true'
#     false_path = base_path / mode / 'false'
#     os.makedirs(base_path, exist_ok=True)
#     os.makedirs(true_path, exist_ok=True)
#     os.makedirs(false_path, exist_ok=True)
#     print(f'Generating dataset {base_path}', dataset, mode)
#     img_data = generate_data(img_num, min_obj_num, max_obj_num, dataset, color_dict)
#
#     # for data_sign in ["true", "false"]:
#     #     data_path = true_path if data_sign == "true" else false_path
#     #     generate_imgs(dataset, img_data, data_path, color_dict)


# def generateClasses(pattern_name, mode, kfgen, n=50, width=200, counterfactual=False):
#     base_path = config.kp_dataset / pattern_name
#     true_path = base_path / mode / 'true'
#     false_path = base_path / mode / 'false'
#     cf_path = base_path / mode / 'counterfactual'
#     os.makedirs(base_path, exist_ok=True)
#     os.makedirs(true_path, exist_ok=True)
#     os.makedirs(false_path, exist_ok=True)
#     os.makedirs(cf_path, exist_ok=True)
#     print(f'Generating dataset {base_path}', pattern_name, mode)
#
#     for (i, kf) in enumerate(kfgen.true_kf(n)):
#         image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
#         data = kf2data(kf, width)
#         with open(true_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
#             json.dump(data, f)
#         image.save(true_path / f"{pattern_name}_{i:06d}.png")
#
#     for (i, kf) in enumerate(kfgen.false_kf(n)):
#         image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
#         data = kf2data(kf, width)
#         with open(false_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
#             json.dump(data, f)
#         image.save(false_path / f"{pattern_name}_{i:06d}.png")
#     if (counterfactual):
#         for (i, kf) in enumerate(kfgen.almost_true_kf(n)):
#             image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
#             data = kf2data(kf, width)
#             with open(cf_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
#                 json.dump(data, f)
#             image.save(cf_path / f"{pattern_name}_{i:06d}.png")


# def generate_data(data_sign, image_num, min_obj_num, max_obj_num, dataset, color_dict):
#     img_data_list = []
#     for i in range(image_num):
#         if dataset == "kp_cha_01":
#             shapeOnshapeObjects = ShapeOnShape(u, 20, 40)
#             generateClasses(dataset, shapeOnshapeObjects, n=image_num, width=512, counterfactual=True)
#         elif dataset == 'kp_line':
#             img_data = patterns.generate_on_line(min_obj_num, max_obj_num, color_dict)
#             while overlaps(img_data):
#                 img_data = patterns.generate_on_line_pair(min_obj_num, max_obj_num, color_dict)
#         elif dataset == "kp_random":
#             img_data = patterns.randomShapes(min_obj_num, max_obj_num, color_dict)
#             while overlaps(img_data):
#                 img_data = patterns.randomShapes(min_obj_num, max_obj_num, color_dict)
#         else:
#             raise ValueError
#         img_data_list.append(img_data)
#         print(f"generate image data: {i}")
#
#     return img_data_list

def kf2data(kf, width):
    data = []
    for obj in kf:
        data.append({"x": obj.x,
                     "y": obj.y,
                     "size": obj.size,
                     "color_name": bk.color_large.index(obj.color),
                     "color_r": bk.color_matplotlib[obj.color][0],
                     "color_g": bk.color_matplotlib[obj.color][1],
                     "color_b": bk.color_matplotlib[obj.color][2],
                     "shape": bk.bk_shapes.index(obj.shape),
                     "width": width
                     })
    return data


def kf2tensor(kf, max_length):
    tensors = []
    for obj in kf:
        color = np.array((bk.color_matplotlib[obj.color])) / 255
        tri = 1 if obj.shape == "triangle" else 0
        sq = 1 if obj.shape == "square" else 0
        cir = 1 if obj.shape == "circle" else 0
        tensor = gestalt_group.gen_group_tensor(obj.x, obj.y, obj.size, 1,
                                                color[0], color[1], color[2], tri, sq, cir)
        tensors.append(tensor)
    if len(tensors) < max_length:
        tensors = tensors + [torch.zeros(len(tensors[0]))] * (max_length - len(tensors))
    else:
        raise ValueError
    tensors = torch.stack(tensors)
    return tensors


def genShapeOnShape(args):
    shapes = bk.bk_shapes[1:]
    width = 512
    size_list = np.arange(0.05, 0.90, 0.05)
    line_width_list = np.arange(0.05, 2, 0.05)
    size_lw = [(x, y) for x in size_list for y in line_width_list]

    for shape in shapes:
        base_path = config.kp_base_dataset / f"{shape}"
        os.makedirs(base_path, exist_ok=True)
        png_num = len([f for f in Path(base_path).iterdir() if
                       f.is_file() and f.suffix == '.png'])
        n = len(size_lw) - png_num  # only generate insufficient ones
        if n <= 0:
            continue
        shapeOnshapeObjects = ShapeOnShape(u, 20, 40)
        for mode in ['train']:
            if shape == "circle":
                gen_fun = shapeOnshapeObjects.cir_only
            elif shape == "triangle":
                gen_fun = shapeOnshapeObjects.tri_only
            elif shape == "square":
                gen_fun = shapeOnshapeObjects.square_only
            elif shape == "gestalt_triangle":
                gen_fun = shapeOnshapeObjects.gestalt_triangle
            elif shape == "diamond":
                gen_fun = shapeOnshapeObjects.dia_only
            elif shape == "trianglecircle":
                gen_fun = shapeOnshapeObjects.triangle_circle
            elif shape == "squarecircle":
                gen_fun = shapeOnshapeObjects.square_circle
            elif shape == "trianglesquare":
                gen_fun = shapeOnshapeObjects.triangle_square
            elif shape == "diamondcircle":
                gen_fun = shapeOnshapeObjects.diamond_circle
            elif shape == "trianglesquarecircle":
                gen_fun = shapeOnshapeObjects.triangle_square_circle
            elif shape == "trianglepartsquare":
                gen_fun = shapeOnshapeObjects.triangle_partsquare
            elif shape == "parttrianglepartsquare":
                gen_fun = shapeOnshapeObjects.parttriangle_partsquare
            elif shape == "random":
                gen_fun = shapeOnshapeObjects.false_kf
            else:
                raise ValueError
            for (i, kf) in enumerate(gen_fun(n, rule_style=False, size_lw=size_lw)):
                image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
                if image is None:
                    continue
                data = kf2data(kf, width)
                with open(base_path / f"{shape}_{(png_num + i):06d}.json", 'w') as f:
                    json.dump(data, f)
                image.save(base_path / f"{shape}_{(png_num + i):06d}.png")


def get_task_names(principle):
    if principle == "good_figure":
        task_names = ["good_figure_two_groups",
                      "good_figure_three_groups",
                      "good_figure_always_three"]
    elif principle == "proximity":
        task_names = ["proximity_red_triangle", ]
    elif principle == "similarity_shape":
        task_names = ["similarity_triangle_circle"]
    elif principle == "similarity_color":
        task_names = ["similarity_two_pairs"]
    elif principle == "closure":
        task_names = ["gestalt_triangle",
                      "gestalt_square",
                      "gestalt_circle",
                      "tri_group",
                      "square_group",
                      "triangle_square"]
    elif principle == "continuity":
        task_names = ["continuity_one_splits_two",
                      "continuity_one_splits_three"]
    elif principle == "symmetry":
        task_names = ["symmetry_pattern"]
    else:
        raise ValueError
    return task_names


def gen_and_save(path, width, mode):
    max_length = 64
    example_num = 3
    all_tensors = {"positive": [], "negative": []}
    task_counter = 0
    principles = bk.gestalt_principles
    for principle in principles:
        task_names = get_task_names(principle)

        for t_i, task_name in enumerate(task_names):
            print("Generating training patterns for task {}".format(task_name))
            img_data = []
            kfs = []
            for dtype in [True, False]:
                for example_i in range(example_num):
                    kfs.append(gestalt_patterns.gen_patterns(task_name, dtype))  # pattern generation
            tensors = []
            images = []
            for kf in kfs:
                img = np.asarray(KandinskyUniverse.kandinskyFigureAsImage(kf, width)).copy()
                images.append(img)
                img_data.append(kf2data(kf, width))
                tensors.append(kf2tensor(kf, max_length))
            tensors = torch.stack(tensors)

            # save image
            os.makedirs(path / ".." / f"{mode}_all", exist_ok=True)
            os.makedirs(path / ".." / f"{mode}_all" / f"{task_counter}", exist_ok=True)
            for img_i in range(len(images)):
                Image.fromarray(images[img_i]).save(
                    path / ".." / f"{mode}_all" / f"{task_counter}" / f"sep_{task_counter:06d}_{img_i}.png")
            images = chart_utils.hconcat_imgs(images)
            Image.fromarray(images).save(path / f"{task_counter:06d}.png")
            # save data
            data = {"principle": principle,
                    "img_data": img_data}
            with open(path / f"{task_counter:06d}.json", 'w') as f:
                json.dump(data, f)

            # save tensor
            all_tensors["positive"].append(tensors[:3])
            all_tensors["negative"].append(tensors[3:])

            task_counter += 1
    return all_tensors


def genGestaltTraining():
    width = 1024
    base_path = config.kp_gestalt_dataset
    os.makedirs(base_path, exist_ok=True)
    for mode in ['train', "test"]:
        data_path = base_path / mode
        os.makedirs(data_path, exist_ok=True)
        tensor_file = data_path / f"{mode}.pt"
        if os.path.exists(tensor_file):
            continue
        tensors = gen_and_save(data_path, width, mode)
        torch.save(tensors, tensor_file)
    print("")


def generate_triangle_image(image_size=512, min_size=20, max_size=200):
    """
    Generate a single image with a random triangle.

    Args:
        image_size (int): Size of the square image (image_size x image_size).
        min_size (int): Minimum size of the triangle.
        max_size (int): Maximum size of the triangle.

    Returns:
        np.ndarray: Generated image as a NumPy array.
        list: List of vertices positions.
    """
    # Create a blank white image
    image = Image.new("L", (image_size, image_size), color="black")
    draw = ImageDraw.Draw(image)

    # Randomly determine the triangle size
    triangle_size = np.random.randint(min_size, max_size)

    # Ensure the triangle is fully within bounds
    margin = 50 + triangle_size // 2
    x_center = np.random.randint(margin, image_size - margin)
    y_center = np.random.randint(margin, image_size - margin)

    # Calculate the vertices of the triangle
    half_height = int(np.sqrt(3) / 2 * triangle_size)
    vertices = [
        (x_center, y_center - 2 * half_height // 3),  # Top vertex
        (x_center - triangle_size // 2, y_center + half_height // 3),  # Bottom-left vertex
        (x_center + triangle_size // 2, y_center + half_height // 3)  # Bottom-right vertex
    ]

    # Random color for the triangle
    color = 255

    # Draw the triangle
    draw.polygon(vertices, fill=color)

    return np.array(image), vertices


def generate_circle_image(image_size=512, min_size=20, max_size=200, hollow=False, max_border_width=10):
    """
    Generate a single image with a random circle.

    Args:
        image_size (int): Size of the square image (image_size x image_size).
        min_size (int): Minimum size of the circle diameter.
        max_size (int): Maximum size of the circle diameter.
        hollow (bool): Whether the circle should be hollow (transparent inside).
        max_border_width (int): Maximum width of the border for hollow circles.

    Returns:
        np.ndarray: Generated image as a NumPy array.
        list: Center and radius of the circle.
    """
    # Create a blank transparent image
    image = Image.new("L", (image_size, image_size), color="black")
    draw = ImageDraw.Draw(image)

    # Randomly determine the circle diameter
    circle_diameter = np.random.randint(min_size, max_size)

    # Ensure the circle is fully within bounds
    margin = 50 + circle_diameter // 2
    x_center = np.random.randint(margin, image_size - margin)
    y_center = np.random.randint(margin, image_size - margin)

    # Define the bounding box for the circle
    bounding_box = [
        (x_center - circle_diameter // 2, y_center - circle_diameter // 2),
        (x_center + circle_diameter // 2, y_center + circle_diameter // 2)
    ]

    # Random color for the circle
    color = 255

    # Draw the filled circle
    draw.ellipse(bounding_box, fill=color)
    # Calculate four random points on the edge of the circle
    angles = np.random.uniform(0, 2 * np.pi, 4)
    edge_points = [
        (x_center + int((circle_diameter // 2) * np.cos(angle)),
         y_center + int((circle_diameter // 2) * np.sin(angle)))
        for angle in angles
    ]
    return np.array(image), edge_points


def generate_square_image(image_size=512, min_size=20, max_size=200, hollow=False, max_border_width=10):
    """
    Generate a single image with a random square.

    Args:
        image_size (int): Size of the square image (image_size x image_size).
        min_size (int): Minimum size of the square.
        max_size (int): Maximum size of the square.
        hollow (bool): Whether the square should be hollow (transparent inside).
        max_border_width (int): Maximum width of the border for hollow squares.

    Returns:
        np.ndarray: Generated image as a NumPy array.
        list: List of vertices positions.
    """
    # Create a blank transparent image
    image = Image.new("L", (image_size, image_size), color="black")
    draw = ImageDraw.Draw(image)

    # Randomly determine the square size
    square_size = np.random.randint(min_size, max_size)

    # Ensure the square is fully within bounds
    margin = 50 + square_size // 2
    x_center = np.random.randint(margin, image_size - margin)
    y_center = np.random.randint(margin, image_size - margin)

    # Calculate the vertices of the square
    vertices = [
        (x_center - square_size // 2, y_center - square_size // 2),  # Top-left vertex
        (x_center + square_size // 2, y_center - square_size // 2),  # Top-right vertex
        (x_center + square_size // 2, y_center + square_size // 2),  # Bottom-right vertex
        (x_center - square_size // 2, y_center + square_size // 2)  # Bottom-left vertex
    ]

    # Random color for the square
    color = 255

    draw.rectangle((vertices[0], vertices[2]), fill=color)

    return np.array(image), vertices


def extract_patches(image, vertices, patch_size=32):
    """
    Extract patches of size patch_size x patch_size around each vertex.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        vertices (list): List of (x, y) tuples representing vertex coordinates.
        patch_size (int): Size of the patch to extract (default is 32).

    Returns:
        list: List of extracted patches as NumPy arrays.
    """
    patches = []
    half_size = patch_size // 2
    padded_image = np.pad(image, ((half_size, half_size), (half_size, half_size)), mode='constant', constant_values=0)

    for (x, y) in vertices:
        x_p, y_p = x + half_size, y + half_size
        patch = padded_image[y_p - half_size:y_p + half_size, x_p - half_size:x_p + half_size]
        patches.append(patch)

    return patches


def generate_dataset(shape, output_dir="triangle_dataset", num_images=1000, image_size=512):
    """
    Generate a dataset of triangle images.

    Args:
        output_dir (str): Directory to save the dataset.
        num_images (int): Number of images to generate.
        image_size (int): Size of the square image (image_size x image_size).
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for i in tqdm(range(num_images)):
        # Generate a triangle image
        if shape == "triangle":
            image, vertices = generate_triangle_image(image_size=image_size)
        elif shape == "square":
            image, vertices = generate_square_image(image_size=image_size)
        elif shape == "circle":
            image, vertices = generate_circle_image(image_size=image_size)
        else:
            raise ValueError

        # Extract patches for each vertex
        patches = extract_patches(image, vertices)

        # Save the image
        image_filename = os.path.join(output_dir, f"triangle_{i:04d}.png")
        image = Image.fromarray(image)
        image.save(image_filename)

        # Save patches
        for j, patch in enumerate(patches):
            patch_filename = os.path.join(output_dir, f"{shape}_{i:04d}_patch_{j}.png")
            patch_image = Image.fromarray(patch)
            patch_image.save(patch_filename)

        # Record metadata
        metadata.append({
            "image": f"triangle_{i:04d}.png",
            "vertices": vertices,
            "patches": [f"{shape}_{i:04d}_patch_{j}.png" for j in range(len(patches))]
        })

        # if (i + 1) % 100 == 0:
        #     print(f"Generated {i + 1}/{num_images} images")

    # Save metadata as JSON
    metadata_filename = os.path.join(output_dir, f"metadata.json")
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to '{metadata_filename}'.")


if __name__ == "__main__":
    num_images_to_generate = 1000
    image_size = 512
    # Parameters
    for shape in ["circle", "square", "triangle"]:
        output_directory = config.kp_base_dataset / shape
        # Generate dataset
        generate_dataset(shape, output_dir=output_directory, num_images=num_images_to_generate, image_size=image_size)
        print(f"Dataset generation complete. Images saved to '{output_directory}'.")
