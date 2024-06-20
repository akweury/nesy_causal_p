# Created by jing at 16.06.24

from utils import visual_utils, file_utils
import config

# data file
f_train_cha = config.data_file_train_cha
f_train_sol = config.data_file_train_sol
raw_data_train = file_utils.get_raw_data(f_train_cha)

data_ids = list(raw_data_train.keys())
first_data = raw_data_train[data_ids[0]]

# show task
demo_train_img = visual_utils.export_task_img(first_data["train"])
demo_test_img = visual_utils.export_task_img(first_data["test"])

#


print("finish!")
# visual_utils.show_array(task_img, "task_01")
