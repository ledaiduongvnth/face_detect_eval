import os
import cv2
from insightface.app import MaskRenderer
from numpy import random

# TODO CHANGE PROCESS ORDER
process_order = 0
num_process = 1
gpu_id = 0
start_folder = 0

input_dataset_dir = "/root/GenMask/WebFace260M"
output_dataset_dir = "/root/GenMask/WebFace260MMask"
# Create output dataset dir
if not os.path.exists(output_dataset_dir):
    os.makedirs(output_dataset_dir)

dirs = os.listdir(input_dataset_dir)
dirs = sorted(dirs)
del dirs[:start_folder]
num_dirs = len(dirs)
num_images_per_task = int(num_dirs / num_process)
job_dir = dirs[process_order * num_images_per_task: (process_order + 1) * num_images_per_task]
print("Total folder:", len(job_dir))

tool = MaskRenderer()
tool.prepare(ctx_id=gpu_id, det_size=(128, 128))
position_style = [[0.1, 0.33, 0.9, 0.7], [0.1, 0.5, 0.9, 0.7]]

masks = ["mask_white", "mask_black", "mask_blue", "mask_green"]

# loop over each folder in the dataset
for i_dir, image_dir in enumerate(job_dir):
    print("Process dir number:", i_dir)
    real_input_image_dir = os.path.join(input_dataset_dir, image_dir)
    real_output_image_dir = os.path.join(output_dataset_dir, image_dir)
    # Create image dir
    if not os.path.exists(real_output_image_dir):
        os.makedirs(real_output_image_dir)

    num_interation = -1
    number_files_in_folder = len(os.listdir(real_input_image_dir))
    if number_files_in_folder <= 5:
        num_interation = int(10/number_files_in_folder)

    # need to check number of images in each folder
    files_path = [f for f in os.listdir(real_input_image_dir) if os.path.isfile(os.path.join(real_input_image_dir, f))]
    # loop over files:
    for file_path in files_path:
        # Write origin image
        real_input_file_path = os.path.join(real_input_image_dir, file_path)
        real_output_file_path = os.path.join(real_output_image_dir, file_path.rstrip(".jpg") + "_" + str(-1) + ".jpg")
        # read image from real_input_file_path, add mask and write to real_output_file_path
        # print("Process:", real_output_file_path)
        image = cv2.imread(real_input_file_path)
        try:
            cv2.imwrite(real_output_file_path, image)
            print("Write to:", real_output_file_path)
        except Exception as e:
            print(e)

        # Gen image
        real_input_file_path = os.path.join(real_input_image_dir, file_path)
        real_output_file_path = os.path.join(real_output_image_dir, file_path)
        # read image from real_input_file_path, add mask and write to real_output_file_path
        # print("Process:", real_output_file_path)
        image = cv2.imread(real_input_file_path)
        mask_index = random.randint(len(masks))
        mask_image = masks[mask_index]
        style_index = 0
        position = position_style[style_index]
        try:
            params = tool.build_params(image)
            mask_out = tool.render_mask(image, mask_image, params, positions=position)
            cv2.imwrite(real_output_file_path, mask_out)
            print("Write to:", real_output_file_path)
        except Exception as e:
            print(e)

        # Gen half mask image
        gen_half_mask = random.randint(5) == 0
        if gen_half_mask:
            real_input_file_path = os.path.join(real_input_image_dir, file_path)
            real_output_file_path = os.path.join(real_output_image_dir, file_path.rstrip(".jpg") + "_" + str(-2) + ".jpg")
            # read image from real_input_file_path, add mask and write to real_output_file_path
            # print("Process:", real_output_file_path)
            image = cv2.imread(real_input_file_path)
            mask_index = random.randint(len(masks))
            mask_image = masks[mask_index]
            style_index = 1
            position = position_style[style_index]
            try:
                params = tool.build_params(image)
                mask_out = tool.render_mask(image, mask_image, params, positions=position)
                cv2.imwrite(real_output_file_path, mask_out)
                print("Write to:", real_output_file_path)
            except Exception as e:
                print(e)

        # Duplicate image
        if num_interation > 0:
            for i in range(num_interation):
                real_input_file_path = os.path.join(real_input_image_dir, file_path)
                real_output_file_path = os.path.join(real_output_image_dir, file_path.rstrip(".jpg") + "_" + str(i) + ".jpg")
                # read image from real_input_file_path, add mask and write to real_output_file_path
                # print("Process:", real_output_file_path)
                image = cv2.imread(real_input_file_path)
                mask_index = random.randint(len(masks))
                mask_image = masks[mask_index]
                style_index = 0
                position = position_style[style_index]
                try:
                    params = tool.build_params(image)
                    mask_out = tool.render_mask(image, mask_image, params, positions=position)
                    cv2.imwrite(real_output_file_path, mask_out)
                    print("Write to:", real_output_file_path)
                except Exception as e:
                    print(e)
