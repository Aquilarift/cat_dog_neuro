import shutil
import os


data_dir = 'C:\\Users\\kelat\Desktop\\neuralnet\\pixel_enhance\\pixel_enhance\\images'

train_dir = 'train'

val_dir = 'val'

test_dir = 'test'

test_data_portion = 0.10

val_data_portion = 0.10

nb_images = 12500


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "cats"))
    os.makedirs(os.path.join(dir_name, "dogs"))


def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "cat." + str(i) + ".jpg"),
                     os.path.join(dest_dir, "cats"))
        shutil.copy2(os.path.join(source_dir, "dog." + str(i) + ".jpg"),
                     os.path.join(dest_dir, "dogs"))


create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)


start_val_data_idx = int(
    nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))


copy_images(0, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)
