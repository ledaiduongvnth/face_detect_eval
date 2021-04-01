TODO :

1, Create a program for making groundtruth bounding box of a given face dataset.
- Using Retinaface pytorch Resnet50 for creating groundtruth bounding box 
https://github.com/biubug6/Pytorch_Retinaface
https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
- Created Ground truth of given face dataset can be used with evaluation.py



GUIDE:

***python3 gen_val_txt.py (to generate txt file that include image path to read and detect) ./widerface_val_TEST/1.txt

***python3 write_mat_file (to detect with specified model that will generate prediction bounding box and mat file) 
        ***prediction folder: ./z
        ***mat file: 1.mat
    ***comment save 1.mat file if you wanna detect with test model
    ***commetn save to prediction model if you wanna gen ground truth file

***python3 evaluation_test.py (to evaluate prediction bouding box with ground truth)

