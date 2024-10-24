from PIL import Image, ImageDraw
import numpy as np
import os
import json
import config
from tqdm import tqdm
from kandinsky_generator.src.kp import KandinskyUniverse
import kandinsky_generator.utils as utils
import kandinsky_generator.patterns as patterns
from kandinsky_generator.ShapeOnShapes import ShapeOnShape
from kandinsky_generator.src.kp.KandinskyUniverse import matplotlib_colors

u = KandinskyUniverse.SimpleUniverse()

###
# Parameters for images: object sizes, object colors, background colors
###
WIDTH = 640

MINSIZE = 10 * 5
MAXSIZE = 24 * 5

# clevr
# kandinsky_colors = {"red": (173, 35, 35),
#                     "yellow": (255, 238, 51), "blue": (42, 75, 215)}

# a bit lighter
# kandinsky_colors = [(193, 85, 85), (255, 238, 101), (90, 135, 235)]
background = (215, 215, 215, 255)


def overlaps(objs):
    image = Image.new("L", (WIDTH, WIDTH), 0)
    sumarray = np.array(image)
    d = ImageDraw.Draw(image)

    for obj in objs:
        image = Image.new("L", (WIDTH, WIDTH), 0)
        draw = ImageDraw.Draw(image)
        utils.draw_obj(draw, obj['shape'], obj['size'], 10, obj["cx"], obj["cy"])

        sumarray += np.array(image)

    sumimage = Image.fromarray(sumarray)
    return sumimage.getextrema()[1] > 10


def kandinskyFigure(objs, color_dict, subsampling=1):
    image = Image.new("RGB", (subsampling * WIDTH, subsampling * WIDTH), background)
    draw = ImageDraw.Draw(image)
    for obj in objs:
        obj_size = subsampling * obj['size']
        obj_color = color_dict[obj["color"]]
        obj_x = subsampling * obj['cx']
        obj_y = subsampling * obj['cy']
        utils.draw_obj(draw, obj['shape'], obj_size, obj_color, obj_x, obj_y)
    if subsampling > 1:
        image = image.resize((WIDTH, WIDTH), Image.BICUBIC)
    return image


def generate_imgs(dataset_name, all_data, save_path, color_dict):
    for d_i, data in enumerate(all_data):
        image = kandinskyFigure(data, color_dict, 1)

        img_data = kandinskyData(data, color_dict, image.width)

        # save image, masks and labels
        image.save(save_path / f"{dataset_name}_{d_i:03d}.png")
        with open(save_path / f"{dataset_name}_{d_i:03d}.json", 'w') as f:
            json.dump(img_data, f)
        print(f"Successfully saved image {d_i}.")


def to_images(kfgen, n=50, width=640):
    pos_imgs = []
    neg_imgs = []
    for (i, kf) in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        pos_imgs.append(image)

    for (i, kf) in enumerate(kfgen.false_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        neg_imgs.append(image)
    return pos_imgs, neg_imgs


def kandinskyData(objs, color_dict, width):
    data = []
    for obj in objs:
        data.append({"x": obj["cx"],
                     "y": obj["cy"],
                     "size": obj['size'],
                     "color_name": obj["color"],
                     "color_r": color_dict[obj["color"]][0],
                     "color_g": color_dict[obj["color"]][1],
                     "color_b": color_dict[obj["color"]][2],
                     "shape": obj["shape"],
                     "width": width
                     })
    return data


def main(dataset, min_obj_num, max_obj_num, color_dict, mode='train', img_num=1):
    # save iamges into true/false folders
    base_path = config.kp_dataset / dataset
    true_path = base_path / mode / 'true'
    false_path = base_path / mode / 'false'
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(true_path, exist_ok=True)
    os.makedirs(false_path, exist_ok=True)
    print(f'Generating dataset {base_path}', dataset, mode)
    img_data = generate_data(img_num, min_obj_num, max_obj_num, dataset, color_dict)

    # for data_sign in ["true", "false"]:
    #     data_path = true_path if data_sign == "true" else false_path
    #     generate_imgs(dataset, img_data, data_path, color_dict)


def kf2data(kf, width):
    data = []
    for obj in kf:
        data.append({"x": obj.x,
                     "y": obj.y,
                     "size": obj.size,
                     "color_name": obj.color,
                     "color_r": matplotlib_colors[obj.color][0],
                     "color_g": matplotlib_colors[obj.color][1],
                     "color_b": matplotlib_colors[obj.color][2],
                     "shape": obj.shape,
                     "width": width
                     })
    return data


def generateClasses(pattern_name, mode, kfgen, n=50, width=200, counterfactual=False):
    base_path = config.kp_dataset / pattern_name
    true_path = base_path / mode / 'true'
    false_path = base_path / mode / 'false'
    cf_path = base_path / mode / 'counterfactual'
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(true_path, exist_ok=True)
    os.makedirs(false_path, exist_ok=True)
    os.makedirs(cf_path, exist_ok=True)
    print(f'Generating dataset {base_path}', pattern_name, mode)

    for (i, kf) in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        data = kf2data(kf, width)
        with open(true_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
            json.dump(data, f)
        image.save(true_path / f"{pattern_name}_{i:06d}.png")

    for (i, kf) in enumerate(kfgen.false_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        data = kf2data(kf, width)
        with open(false_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
            json.dump(data, f)
        image.save(false_path / f"{pattern_name}_{i:06d}.png")
    if (counterfactual):
        for (i, kf) in enumerate(kfgen.almost_true_kf(n)):
            image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
            data = kf2data(kf, width)
            with open(cf_path / f"{pattern_name}_{i:06d}.json", 'w') as f:
                json.dump(data, f)
            image.save(cf_path / f"{pattern_name}_{i:06d}.png")


def generate_data(data_sign, image_num, min_obj_num, max_obj_num, dataset, color_dict):
    img_data_list = []
    for i in range(image_num):
        if dataset == "kp_cha_01":
            shapeOnshapeObjects = ShapeOnShape(u, 20, 40)
            generateClasses(dataset, shapeOnshapeObjects, n=image_num, width=512, counterfactual=True)
        elif dataset == 'kp_line':
            img_data = patterns.generate_on_line(min_obj_num, max_obj_num, color_dict)
            while overlaps(img_data):
                img_data = patterns.generate_on_line_pair(min_obj_num, max_obj_num, color_dict)
        elif dataset == "kp_random":
            img_data = patterns.randomShapes(min_obj_num, max_obj_num, color_dict)
            while overlaps(img_data):
                img_data = patterns.randomShapes(min_obj_num, max_obj_num, color_dict)
        else:
            raise ValueError
        img_data_list.append(img_data)
        print(f"generate image data: {i}")

    return img_data_list


def genShapeOnShape(shape, n):
    shapeOnshapeObjects = ShapeOnShape(u, 20, 40)
    for mode in ['train']:
        width = 512
        base_path = config.kp_dataset / f"data_{task}"
        true_path = base_path / mode / 'true'
        false_path = base_path / mode / 'false'
        cf_path = base_path / mode / 'counterfactual'
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(true_path, exist_ok=True)
        os.makedirs(false_path, exist_ok=True)
        os.makedirs(cf_path, exist_ok=True)
        print(f'Generating dataset {base_path}', task, mode)
        if shape == "circle":
            gen_fun = shapeOnshapeObjects.cir_only
        elif shape == "diamond":
            gen_fun = shapeOnshapeObjects.dia_only
        elif shape == "triangle":
            gen_fun = shapeOnshapeObjects.tri_only
        elif shape == "square":
            gen_fun = shapeOnshapeObjects.square_only
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

        for (i, kf) in tqdm(enumerate(gen_fun(n))):
            image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
            data = kf2data(kf, width)
            with open(true_path / f"{task}_{i:06d}.json", 'w') as f:
                json.dump(data, f)
            image.save(true_path / f"{task}_{i:06d}.png")


if __name__ == '__main__':
    # task = "kp_cha_01"
    tasks = ["diamondcircle"]
    for task in tasks:
        genShapeOnShape(task, 100)
