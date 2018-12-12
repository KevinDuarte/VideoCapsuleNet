import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# batch size and number of epochs
batch_size = 8
n_epochs = 120

# number of epochs to train in between validations
n_eps_for_eval = 3

# training accuracy threshold needed for validation to run
acc_for_eval = 0.9

# number of epochs until validation can start
n_eps_until_eval = 40

# learning rate and beta1 are used in the Adam optimizer.
learning_rate, beta1 = 0.0001, 0.5

# Used to prevent numerical instability (dividing by zero or log(0))
epsilon = 1e-6

use_c3d_weights = True

# number of classes for the network to predict
n_classes = 24

# model number, output file name, save file directory, and save file name
model_num = 2
output_file_name = './output%d.txt' % model_num
network_save_dir = './network_saves/'
if not os.path.exists(network_save_dir):  # creates the directory if it does not exist
    os.mkdir(network_save_dir)
save_file_name = network_save_dir + ('model_%d.ckpt' % model_num)

# coefficient for the segmentation loss
segment_coef = 0.0002

# margin for classification loss, how much it is incremented by, and how often it is incremented by
start_m = 0.2
m_delta = 0.1
n_eps_for_m = 5

# number of frames to skip in the data
frame_skip = 2

# time to wait for data to load when dataloader is created
wait_for_data = 5

# number of batches to train on before statistics are printed to stdio
batches_until_print = 100

# parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1

# size of the pose matrix height and width
pose_dimension = 4

# determines if the network layers will be printed when network is initialized
print_layers = True


def clear_output():
    """
    Clears the text file which the training/validation/testing metrics will be printed to
    """
    with open(output_file_name, 'w') as f:
        print('Writing to ' + output_file_name)


def write_output(string):
    """
    Writes a given string to the text output file. Used to write the different metrics.
    """
    try:
        output_log = open(output_file_name, 'a')
        output_log.write(string)
        output_log.close()
    except:
        print('Unable to save to output log')
