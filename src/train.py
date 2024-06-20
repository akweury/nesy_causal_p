# Created by jing at 17.06.24


from utils import visual_utils, file_utils, args_utils
import config
import grouping

# arguments

args = args_utils.get_args()

args.demo_id = 4

# data file
f_train_cha = config.data_file_train_cha
f_train_sol = config.data_file_train_sol
raw_data_train = file_utils.get_raw_data(f_train_cha)

data_ids = list(raw_data_train.keys())

for id in range(len(data_ids)):
    args.demo_id = id
    first_data = raw_data_train[data_ids[args.demo_id]]
    patch_input, patch_output = grouping.data2patch(args, first_data['train'])

    # ------ actions --------
    # duplicate: 0, 4
    # fill: 1
    # cyclic: 2,6
    # stretch: 3
    # override: 5
    # shift: 7
    # connection: 8
    # coloring: 9, 10

    # ------ recognize groups --------
    # identical input in output:
    #   - same color: 0, 1
    #   - different color: 2
    # regions with borders: 5,10
    # connection:
    #  - same color: 3,4,6,7,8,9,12
    #  - different color: 11,

    print(f"task id: {args.demo_id}/{len(data_ids)}")
    visual_utils.export_task_img(args, patch_input, patch_output)
    # id_groups, patch_groups = grouping.group_with_identical(args, patch_input, patch_output)
    # visual_utils.visual_reasoning(args, patch_input, patch_output, patch_groups, id_groups)
print('finish!')
