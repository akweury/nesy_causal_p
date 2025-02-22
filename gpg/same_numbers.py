# Created by J at 18.02.25
import random
import math
import numpy as np

from kandinsky_generator.src.kp.KandinskyUniverse import kandinskyShape


def generate_group_objects(color, target_count, cluster_centers, image_size, diameter,
                           so, min_circles, max_circles, used_centers):
    """Generate objects for one group (of a given color) until target_count is reached."""
    group_objs = []
    count = 0
    while count < target_count:
        # Pick a random cluster center to start a new cluster.
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_points = [(cluster_x, cluster_y)]
        used_centers.add((cluster_x, cluster_y))

        # Decide how many circles to generate in this cluster.
        num_circles = np.random.randint(min_circles, max_circles + 1)

        # Create additional circles in the cluster.
        for _ in range(num_circles - 1):
            if count >= target_count:
                break
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
            ]
            random.shuffle(directions)  # Try different directions at random
            for dx, dy in directions:
                new_x = cluster_points[-1][0] + dx
                new_y = cluster_points[-1][1] + dy
                # Check that the new circle is within bounds and not overlapping existing ones.
                if (0.05 < new_x < image_size[0] and 0.05 < new_y < image_size[1] and
                        all((new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in used_centers)):
                    cluster_points.append((new_x, new_y))
                    used_centers.add((new_x, new_y))
                    group_objs.append(
                        kandinskyShape(color=color, shape="circle", size=so,
                                       x=new_x, y=new_y, line_width=-1, solid=True)
                    )
                    count += 1
                    break
    return group_objs


def generate_scene(so, dtype, g_num, grid_size=3, min_circles=3, max_circles=5,
                   diameter=0.08, image_size=(1, 1)):
    """
    Generate a scene by creating multiple groups.

    Parameters:
      so           : size parameter for kandinskyShape.
      group_configs: list of tuples (color, target_count) for each group.
      grid_size    : defines the number of cluster centers (grid_size x grid_size).
      min_circles, max_circles: control the number of circles per cluster.
      diameter     : the spacing used for circle placement.
      image_size   : tuple defining the dimensions of the image.

    Returns:
      List of kandinskyShape objects.
    """
    if g_num == 2:
        base_count = random.randint(10, 20)
        if dtype:
            target_yellow = base_count
            target_blue = base_count
        else:
            # When dtype is False, adjust one group’s count to be different.
            if random.random() < 0.5:
                target_yellow = max(1, base_count + random.randint(-5, -1))
            else:
                target_yellow = max(1, base_count + random.randint(1, 5))
            target_blue = base_count
        # Define two groups: yellow and blue.
        group_configs = [("yellow", target_yellow), ("blue", target_blue)]
    elif g_num == 3:
        base_count = random.randint(6, 12)
        if dtype:
            target_yellow = base_count
            target_blue = base_count
            target_red = base_count
        else:
            # When dtype is False, adjust one group’s count to be different.
            if random.random() < 0.5:
                target_yellow = max(1, base_count + random.randint(-5, -1))
            else:
                target_yellow = max(1, base_count + random.randint(1, 5))
            target_blue = base_count
            target_red = max(1, base_count + random.randint(-5, 5))
        # Define two groups: yellow and blue.
        group_configs = [("yellow", target_yellow), ("blue", target_blue), ("red", target_red)]
    elif g_num == 4:
        base_count = random.randint(5, 10)
        if dtype:
            target_yellow = base_count
            target_blue = base_count
            target_red = base_count
            target_green = base_count
        else:
            # When dtype is False, adjust one group’s count to be different.
            if random.random() < 0.5:
                target_yellow = max(1, base_count + random.randint(-3, -1))
            else:
                target_yellow = max(1, base_count + random.randint(1, 3))
            target_blue = base_count
            target_red = max(1, base_count + random.randint(-3, 3))
            target_green = max(1, base_count + random.randint(1, 2))
        # Define two groups: yellow and blue.
        group_configs = [("yellow", target_yellow), ("blue", target_blue), ("red", target_red), ("green", target_green)]

    else:
        raise ValueError("g_num must be 2 or 3 or 4.")

    used_centers = set()
    objs = []
    # Define evenly spaced cluster centers over the image.
    grid_spacing = image_size[0] / (grid_size + 1)
    cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1))
                       for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cluster_centers)

    # Generate objects for each group.
    for color, target_count in group_configs:
        objs.extend(generate_group_objects(color, target_count, cluster_centers,
                                           image_size, diameter, so,
                                           min_circles, max_circles, used_centers))

    return objs

#
# # Your original function can now be written as a special case with two groups.
# def similarity_fixed_number_two(so, dtype, grid_size=3, min_circles=3,
#                                 max_circles=5, diameter=0.1, image_size=(1, 1)):
#     # Override diameter as in the original code.
#     diameter = 0.08
#     base_count = random.randint(10, 20)
#     if dtype:
#         target_yellow = base_count
#         target_blue = base_count
#     else:
#         # When dtype is False, adjust one group’s count to be different.
#         target_yellow = max(1, base_count + random.randint(-5, 5))
#         target_blue = base_count
#     # Define two groups: yellow and blue.
#     group_configs = [("yellow", target_yellow), ("blue", target_blue)]
#     return generate_scene(so, group_configs, grid_size, min_circles, max_circles,
#                           diameter, image_size)
