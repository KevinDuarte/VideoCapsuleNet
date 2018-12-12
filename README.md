## VideoCapsuleNet

This is the code for the NeurIPS 2018 paper VideoCapsuleNet: A Simplified Network for Action Detection. 

The paper can be found here: http://papers.nips.cc/paper/7988-videocapsulenet-a-simplified-network-for-action-detection 

The network is implemented using TensorFlow.

## Data Used

We have supplied the code for training and testing the model on the UCF-101 dataset. The file <code>load_ucf101_data.py</code> creates two DataLoaders - one for training and one for testing. The <code>dataset_dir</code> variable at the top of the file should be set to the base directory which contains the frames and annotations..

To run this code, you need to do the following:
1. Download the UCF-101 dataset at http://crcv.ucf.edu/data/UCF101.php 
a) Extract the frames from each video (downsized to 160x120), and store them as .jpeg files, with the names "frame_K.jpg" where K is the frame number, from 0 to T-1. The path to the frames should be: <code>[dataset_dir]/UCF101_Frames/[Video Name]/frame_K.jpg</code>.
2. Download the trainAnnot.mat and testAnnot.mat Annotations from https://github.com/gurkirt/corrected-UCF101-Annots and the path to the annotations should be <code>[dataset_dir]/UCF101_Annotations/*.mat</code>

## Training and Testing the Model

Once the data is set up you can train (and test) the network by calling <code>python3 caps_main.py</code>.

To get similar results found in the paper, the pretrained C3D weights are needed (see <code>readme.txt</code>) in the pretrained_weights folder.

The <code>config.py</code> file contains several hyper-parameters which are useful for training the network. 

## Output File

During training and testing, metrics are printed to stdout as well as an output*.txt file. During training/validation, the losses and accuracies are printed out. At test time, the accuracy, f-mAP and v-mAP scores (for many IoU thresholds), and f-AP@IoU=0.5 and v-AP@IoU=0.5 for each class, are printed out.

An example of this is found in <code>output2.txt</code>. These are not the same results as those found in the paper (since cleaning the code led to different variable names, so using the same weights would be difficult to transfer) but they are comparable.

## Saved Weights

As the network is trained, the best weights are being saved to the network_saves folder. The weights for the network trained on UCF-101 can be found [here](https://drive.google.com/file/d/1irmiwT9Mt-y5Yr5Kcv5hk8nFizH6N5nL/view?usp=sharing). Unzip the file and place the three .ckpt files in the network_saves folder. These weights correspond the the results found in <code>output2.txt</code>.
