# Created by jing at 16.06.24

import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import IPython.display as display
import tempfile

import config
from src.utils import data_utils


def tile_zoom_in(tile_array, zoom_factor, tile_border=True, lbw=0, rbw=0, tbw=0, bbw=0):
    tile_pad_width = config.tile_pad_width
    all_tiles = []
    for r_i in range(tile_array.shape[0]):
        row_tiles = []
        for c_i in range(tile_array.shape[1]):
            tile = np.zeros((zoom_factor, zoom_factor, 3), dtype=np.uint8) + tile_array[r_i, c_i].tolist()

            # add tile border
            if tile_border:
                tile = np.pad(tile,
                              pad_width=((tile_pad_width, tile_pad_width), (tile_pad_width, tile_pad_width), (0, 0)),
                              constant_values=255)

            row_tiles.append(tile)
        row_img = hconcat_resize(row_tiles)
        all_tiles.append(row_img)

    img = vconcat_resize(all_tiles)
    img = np.pad(img, pad_width=((tbw, bbw), (lbw, rbw), (0, 0)), constant_values=255).astype(np.uint8)
    return img


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def show_array(array, title):
    if array.shape[2] == 3:
        # rgb image
        cv.imshow(f"numpy_{title}", array)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print(f"Unsupported input array shape.")


def uncompress_data(data):
    row_num = len(data)
    col_num = len(data[0])
    data_array = torch.zeros((row_num, col_num, 3), dtype=torch.uint8)
    for i in range(row_num):
        for j in range(col_num):
            data_array[i, j] = torch.tensor(config.color_tiles[data[i][j]], dtype=torch.uint8)
    return data_array


# https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
def concat_vh(list_2d):
    """
    show image in a 2d array
    :param list_2d: 2d array with image element
    :return: concatenated 2d image array
    """
    # return final image
    return cv.vconcat([cv.hconcat(list_h)
                       for list_h in list_2d])


def vconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    w_min = min(img.shape[1] for img in img_list)
    im_list_resize = [cv.resize(img,
                                (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
                      for img in img_list]
    return cv.vconcat(im_list_resize)


def hconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation)
                      for img in img_list]

    return cv.hconcat(im_list_resize)


def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int32(patch_width / 2.0)
    return cv.copyMakeBorder(img, offset, offset, offset, offset, cv.BORDER_CONSTANT,
                             value=(255, 255, 255))


def visual_patch(patch, zoom_factor=1):
    patch_array = uncompress_data(patch)
    return patch_array


def save_image(final_image, image_output_path):
    cv.imwrite(image_output_path, final_image)


def export_task_img(data, output_path):
    """ reasoning process visualization as a gif image.
    From left to right: input, output, reasoning_process."""
    data = list(data.values())
    total_data_num = len(data)
    for id in tqdm(range(total_data_num), desc=f"exporting data to {output_path}"):
        task = data[id]
        patch_input, patch_output = data_utils.data2patch(task['train'])
        task_img = get_task_img(patch_input, patch_output)
        task_img_file = output_path / f'{id:03d}.png'
        # save task img
        save_image(task_img, str(task_img_file))


def visual_horizontal(patch_imgs, img_name):
    img = hconcat_resize(patch_imgs)
    show_array(img, img_name)
    return img


def visual_vertical(img_list, img_name):
    img = vconcat_resize(img_list)
    show_array(img, img_name)
    return img


def highlight_patch(big_patch, small_patch, pos):
    group_img = np.zeros_like(big_patch)
    group_region = big_patch[pos[0]:pos[0] + len(small_patch[0]), pos[1]:pos[1] + len(small_patch)]
    group_img[pos[0]:pos[0] + len(small_patch[0]), pos[1]:pos[1] + len(small_patch)] = group_region

    return group_img


def addText(img, text, pos, font_size=2, color=(255, 255, 255), thickness=2):
    h, w = img.shape[:2]
    position = pos

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


def patch2img(big_patch):
    patch_img = visual_patch(big_patch)

    # zoom the image
    zoom_factor = 512 // max(patch_img.shape[:2])
    zoom_img = tile_zoom_in(patch_img, zoom_factor)

    return zoom_img


def padding_img(img, tbw=0, bbw=0, lbw=0, rbw=0):
    img = np.pad(img, pad_width=((tbw, bbw), (lbw, rbw), (0, 0)), constant_values=255).astype(np.uint8)
    return img


def visual_identical_groups(examples):
    visual_imgs = []
    for example in examples:
        if len(example['input']) < len(example['output']):
            big_patch = example['output']
            small_patch = example['input']
        elif len(example['input']) > len(example['output']):
            big_patch = example['input']
            small_patch = example['output']
        else:
            raise NotImplementedError
        patch_imgs = []
        # add large image
        big_patch_img = patch2img(big_patch)
        patch_imgs.append(big_patch_img)
        # add group images
        for group in example['identical_groups'][:5]:
            big_patch = np.array(big_patch)
            patch_array = highlight_patch(big_patch, small_patch, group['position'])
            # convert patch to image
            group_img = patch2img(patch_array)
            group_img = padding_img(group_img, lbw=50, rbw=200, tbw=50, bbw=200)
            addText(group_img, f"sim:{group['similarity']}", color=(0, 0, 255), pos=(100, 700))
            patch_imgs.append(group_img)
        visual_imgs.append(hconcat_resize(patch_imgs))

    visual_vertical(visual_imgs, "find_identical_groups")


def img_processing(img, lbw=0, rbw=0, tbw=0, bbw=0, text=None):
    img = padding_img(img, lbw=lbw, rbw=rbw, tbw=tbw, bbw=bbw)
    if text is not None:
        addText(img, text, color=(255, 0, 0), pos=(100, 700))
    return img


def get_reasoning_frame(patch_input, patch_output, patch_group, id_groups, g_i):
    frame_img = []
    for e_i in range(len(patch_input)):
        input_img = patch2img(patch_input[e_i])
        input_img = img_processing(input_img, lbw=50, rbw=200, tbw=50, bbw=200, text='In')
        output_img = patch2img(patch_output[e_i])
        output_img = img_processing(output_img, lbw=50, rbw=200, tbw=50, bbw=200, text='Out')
        group_img = patch2img(patch_group[e_i])
        group_img = img_processing(group_img, lbw=50, rbw=200, tbw=50, bbw=200,
                                   text=f"sim:{id_groups[e_i][g_i]['similarity']:.2f}")
        example_img = [input_img, output_img, group_img]
        example_img = hconcat_resize(example_img)
        frame_img.append(example_img)
    frame_img = vconcat_resize(frame_img)
    return frame_img


def get_task_img(patch_input, patch_output):
    task_img = []
    for e_i in range(len(patch_input)):
        input_img = patch2img(patch_input[e_i])
        input_img = img_processing(input_img, lbw=50, rbw=200, tbw=50, bbw=200, text='In')
        output_img = patch2img(patch_output[e_i])
        output_img = img_processing(output_img, lbw=50, rbw=200, tbw=50, bbw=200, text='Out')
        example_img = [input_img, output_img]
        example_img = hconcat_resize(example_img)
        task_img.append(example_img)
    task_img = vconcat_resize(task_img)
    return task_img


def release_video(frames, file):
    num_frames = 30
    height, width = frames[0].shape[0], frames[0].shape[1]  # Define the height and width of the frames
    num_frames = 30
    # height, width = 100, 100  # Define the height and width of the frames
    # frames = [np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    # Create a temporary file to save the video
    video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

    # Define the codec and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 3  # Frames per second
    out = cv.VideoWriter(str(file), fourcc, fps, (width, height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    display.Video(str(file), embed=True)


def visual_reasoning(args, patch_input, patch_output, patch_groups, id_groups):
    """ reasoning process visualization as a gif image.
    From left to right: input, output, reasoning_process."""
    frames = []
    for g_i in range(len(patch_groups[0])):
        p_group_e_i = [patch_groups[ei][g_i] for ei in range(len(patch_groups))]
        frame = get_reasoning_frame(patch_input, patch_output, p_group_e_i, id_groups, g_i)
        frames.append(frame)

    # Convert frames to a video
    release_video(frames, config.output / f'reasoning_{args.demo_id}.mp4')


def group2patch(whole_patch, group):
    data = np.array(whole_patch)
    group_patch = np.zeros_like(data) + 10
    for pos in group:
        group_patch[pos] = data[pos]

    return group_patch


def align_white_imgs(list_a, list_b):
    white_img = np.zeros_like(list_a[0]) + 255
    if len(list_a) < len(list_b):
        align_num = len(list_b) - len(list_a)
        list_a += [white_img] * align_num
    else:
        align_num = len(list_a) - len(list_b)
        list_b += [white_img] * align_num
    return list_a, list_b
