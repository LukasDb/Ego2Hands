import models.CSM.CSM_tf as CSM
from utils import visualize_seg_detection
import cv2
from data_loaders.Ego2Hands_tf import Ego2HandsData
import yaml
import os
from utils import Config
import numpy as np


def main():
    config_path = os.path.join("configs", "config_tf.yml")
    config = Config(config_path)

    seq = 1

    from train_tf import train

    model = CSM.CSM_baseline(n_classes = config.num_classes, with_energy = config.energy, input_edge = False)

    #model.build(input_shape = (config.batch_size, 288, 512, 3))

    #model.summary()

    train(config, model)

    #data = Ego2HandsData(config, "test_seg", seq_i = seq)
    #for idx, (img_real_orig_tensor, img_real_test_tensor, seg_gt_tensor, box_l_tensor, box_r_tensor) in enumerate(data):
        #visualize(idx, seq, img_real_orig_tensor, img_real_test_tensor, seg_gt_tensor, box_l_tensor, box_r_tensor)
        #if idx>5:
            #break


    # data = Ego2HandsData(config, "train_seg", seq_i = 1)
# 
    # model = CSM.CSM_baseline(n_classes = config.num_classes, with_energy = False, input_edge = False)
# 
# 
    # for idx, (img_id, img_real_orig_tensor, img_real_tensor, seg_real_tensor) in enumerate(data):
        # visualize(idx, seq, img_real_orig_tensor, img_real_tensor, seg_real_tensor)
        #if idx>5:
            #break




def visualize(i, seq_i, img_orig_tensor, img_test_tensor, seg_output_final, box_l_gt_tensor=None, box_r_gt_tensor=None):
    img_orig_np = img_orig_tensor.numpy()
    img_test_np = img_test_tensor.numpy().transpose(1,2,0) # channels last
    seg_output_np = seg_output_final.numpy()

    if box_l_gt_tensor is not None:
        box_l_gt_np = box_l_gt_tensor.numpy()
    if box_r_gt_tensor is not None:
        box_r_gt_np = box_r_gt_tensor.numpy()
    
    custom_status = seq_i

    close_kernel_size = 7

    #for batch_i, (img_orig_i, img_test_i, seg_output_i, box_l_gt_i, box_r_gt_i) in enumerate(zip(img_orig_np, img_test_np, seg_output_np, box_l_gt_np, box_r_gt_np)):

    cv2.imshow("img_orig_i", img_orig_np.astype(np.uint8))
    cv2.imshow("seq_img_grayscale.png", ((img_test_np*256+128.0)).astype(np.uint8))
    seg_output_idx = seg_output_np.astype(np.uint8)
    cv2.imshow("seq_output.png", seg_output_idx*50)
    img_vis = visualize_seg_detection(img_orig_np, seg_output_idx, None, None, box_gt_l = None, box_gt_r = None, close_kernel_size = close_kernel_size)
    cv2.imshow("seq_img_vis.png", img_vis.astype(np.uint8))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()