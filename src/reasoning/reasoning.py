# Created by shaji at 25/07/2024

def obj_necessity(task_objs):
    example_num = len(task_objs)
    necessity = {
        "ig_line": 0,
        "og_line": 0,
        "ig_rect": 0,
        "og_rect": 0,
    }
    for e_i in range(len(task_objs)):
        for ig in task_objs[e_i]["input_groups"]:
            if ig["is_line"]:
                necessity["ig_line"] += 1
            if ig["is_rect"]:
                necessity["ig_rect"] += 1
        for og in task_objs[e_i]["output_groups"]:
            if og["is_line"]:
                necessity["og_line"] += 1
            if og["is_rect"]:
                necessity["og_rect"] += 1

    necessity["ig_line"] /= example_num
    necessity["og_line"] /= example_num
    necessity["ig_rect"] /= example_num
    necessity["og_rect"] /= example_num
    return necessity
