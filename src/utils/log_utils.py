# Created by shaji at 24/06/2024

import datetime
def create_log_file(exp_output_path):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = str(exp_output_path / f"log_{date_now}_{time_now}.txt")
    with open(file_name, "w") as f:
        f.write(f"Log ({date_now}, {time_now})")

    return str(exp_output_path / file_name)


def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(str(line_str) + "\n")