# Created by shaji at 25/07/2024

def obj_necessity(task_objs):
    example_num = len(task_objs)
    necessity = {
        "ig": {"line": 0, "rect": 0},
        "og": {"line": 0, "rect": 0},
    }
    for e_i in range(len(task_objs)):
        for ig in task_objs[e_i]["ig"]:
            if len(ig["line"]) > 0:
                necessity["ig"]["line"] += 1
            if len(ig["rect"]) > 0:
                necessity["ig"]["rect"] += 1
        for og in task_objs[e_i]["og"]:
            if len(og["line"]) > 0:
                necessity["og"]["line"] += 1
            if len(og["rect"]) > 0:
                necessity["og"]["rect"] += 1

    necessity["ig"]["line"] /= example_num
    necessity["og"]["line"] /= example_num
    necessity["ig"]["rect"] /= example_num
    necessity["og"]["rect"] /= example_num
    return necessity


def find_group_relation(ig, og, necessity):
    relation = []
    for ig_type in ["line", "rect"]:
        for og_type in ["line", "rect"]:
            if necessity["ig"][ig_type] == 1 and necessity["og"][og_type] == 1:
                if len(ig[ig_type]) > 0 and len(og[og_type]) > 0:
                    relation.append([ig_type, og_type])

    return relation
