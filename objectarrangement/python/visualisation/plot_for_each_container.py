import os
import sys
import shutil
from ae_loss_AE import plot_loss_in_dir_AE
from ae_loss_VAE import plot_loss_in_dir_VAE
from latent_space import plot_latent_space_in_dir
from latent_density import plot_latent_density_in_dir
from pos_var import plot_pos_var_in_dir
from pct_move import plot_pct_move_in_dir

current_path = os.getcwd()
sys.path.append("/git/sferes2/exp/imagesd/python/visualisation")

GENERATE_EACH_IMAGE = True

path = sys.argv[1]
application = sys.argv[2]
variant = application.split("_")[-1]
is_full_loss = "true" in sys.argv[3]

print(f"PROCESSING VISUALISATIONS - {path}")
plot_pos_var_in_dir(path, GENERATE_EACH_IMAGE)
plot_pct_move_in_dir(path, GENERATE_EACH_IMAGE)
plot_latent_density_in_dir(path, GENERATE_EACH_IMAGE)
plot_latent_space_in_dir(path)

# PID level plotting
if variant == "vae":
    plot_loss_in_dir_VAE(path, is_full_loss, GENERATE_EACH_IMAGE)
else:
    plot_loss_in_dir_AE(path, GENERATE_EACH_IMAGE)

os.chdir(path)
os.makedirs("plots", exist_ok=True)
image_files = [img for img in os.listdir() if ".png" in img]
for image in image_files:
    shutil.move(image, f"plots/{image}")

os.chdir(current_path)