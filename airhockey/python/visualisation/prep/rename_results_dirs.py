import os
variant = "aurora"

BASE_NAME = "/home/andwang1/airl/imagesd/test_results/new_traj_stat"

os.chdir(f"{BASE_NAME}/results_imagesd_{variant}")
dir_names = os.listdir()

for name in dir_names:
    if "--" not in name:
        continue

    components = name.split("_")
    args = [i.split("=")[-1] for i in components]
    new_name = f"gen{args[0]}_random{args[1]}_fullloss{args[2]}_beta{args[3]}_extension{args[4]}_lossfunc{args[5]}_sigmoid{args[6]}_sample{args[7]}"

    os.rename(name, new_name)
