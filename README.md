## VideoCapsuleNet

This is the code for the paper VideoCapsuleNet: A Simplified Network for Action Detection, NeurIPS 2018. 

The paper can be found here: http://papers.nips.cc/paper/7988-videocapsulenet-a-simplified-network-for-action-detection 

## Data Used

We have supplied the code for training and testing the model on the UCF-101 dataset. The file <code>load_ucf101_data.py</code> creates two DataLoaders - one for training and one for testing.

To run this code, you need to do the following:
1. Download the UCF-101 dataset at http://crcv.ucf.edu/data/UCF101.php 
a) Extract the frames from each video (downsized to 160x120), and store them as .jpeg files, with the names "frame_\*.jpg" where * is the frame number from 0 to T-1.
2. Download the Annitations
