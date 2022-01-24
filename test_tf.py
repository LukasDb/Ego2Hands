from utils import visualize_seg_detection
import numpy as np
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import AverageMeter
import cv2
from models.CSM.CSM_tf import CSM_TF
import cv2
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import numpy as np
import time


def test_tf(model: CSM_TF, config: Config, seq_i):

    # train
    hand_dataset_test = lambda : Ego2HandsData(config, mode = "test_seg", seq_i = seq_i)
    #elf.img_h, self.img_w = 
    temp = hand_dataset_test()
    h, w = (temp.img_h, temp.img_w) # 288, 512
    
    
    #output:  [img_real_orig_tensor, img_real_test_tensor, seg_gt_tensor]
    output_signature = (
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 2 if config.energy else 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w), dtype=tf.uint8),
    )
    #[print(x.dtype) for x in next(temp)]
    #[print(x.shape) for x in next(temp)]

    test_loader = tf.data.Dataset.from_generator(hand_dataset_test, output_signature=output_signature) \
        .cache().batch(1).prefetch(tf.data.AUTOTUNE)

    if config.save_outputs:
        out_seg_path = "outputs/{}_{}_edge{}_energy{}_seg_test/".format(config.dataset, config.model_name, int(config.input_edge), int(config.energy))
        if not os.path.exists(out_seg_path):
            print("Created outputs directory at {}".format(out_seg_path))
            os.makedirs(out_seg_path)
        
    iou_meter = AverageMeter()
    ap_meter = AverageMeter()
    inf_time_meter = AverageMeter()


    for i, (img_orig_tensor, img_test_tensor, seg_gt_tensor) in enumerate(test_loader):
        img_batch_size = img_test_tensor.shape[0]
        img_h, img_w = img_test_tensor.shape[1], img_test_tensor.shape[2]
        
        if config.speed_test:
            start_time = time.perf_counter()
        
        # Forward pass
        if config.energy:
            if model.n_stages == 1:
                seg_output_final, energy_output_final = model(img_test_tensor)
            elif model.n_stages == 2:
                seg_output_final, energy_output_final = model(img_test_tensor)
            else:
                    _, _, seg_output_final, energy_output_final = model(img_test_tensor)
            energy_l_final = energy_output_final[:,1,:,:]
            energy_r_final = energy_output_final[:,2,:,:]
        else:
            if model.n_stages == 1:
                seg_output_final = model(img_test_tensor)
            elif model.n_stages == 2:
                seg_output_final = model(img_test_tensor)
            else:
                _, seg_output_final = model(img_test_tensor)
            energy_l_final = tf.zeros((img_batch_size, 1, 1, 1))
            energy_r_final = tf.zeros((img_batch_size, 1, 1, 1))

        if config.speed_test:
            end_time = time.perf_counter()
            inf_time_meter.update(end_time - start_time, 1)
        else:
            # Evaluation
            close_kernel_size = 7
            
            #seg_output_final = nn.functional.interpolate(seg_output_final, size = (img_h, img_w), mode = 'bilinear', align_corners = True)
            seg_output_final = tf.image.resize(seg_output_final, size = (img_h, img_w))

            #iou_np = compute_iou(seg_output_final, seg_gt_tensor)
            #iou_meter.update(np.mean(iou_np), 1)
            
            if config.energy:
                energy_l_final = tf.image.resize(tf.expand_dims(energy_l_final, 0), size = (img_h, img_w))
                energy_r_final = tf.image.resize(tf.expand_dims(energy_r_final, 0), size = (img_h, img_w))
            
                #ap_np = compute_ap(energy_l_final.numpy(), energy_r_final.cpu().data.numpy(), box_l_gt_tensor.cpu().data.numpy(), box_r_gt_tensor.cpu().data.numpy(), close_kernel_size = close_kernel_size)
                #ap_meter.update(np.mean(ap_np), 1)
        
            # Visualize Outputs
            if config.save_outputs:
                img_orig_np = img_orig_tensor.numpy()
                img_test_np = img_test_tensor.numpy()
                seg_output_np = seg_output_final.numpy()
                energy_l_np = energy_l_final.numpy()
                energy_r_np = energy_r_final.numpy()
                
                custom_status = seq_i

                for batch_i, (img_orig_i, img_test_i, seg_output_i, energy_l_i, energy_r_i) in enumerate(zip(img_orig_np, img_test_np, seg_output_np, energy_l_np, energy_r_np)):
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_orig.png".format(custom_status, i)), img_orig_i.astype(np.uint8))
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_grayscale.png".format(custom_status, i)), ((img_test_i[:,:,0]*256+128.0)).astype(np.uint8))
                    if config.input_edge:
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_edge.png".format(custom_status, i)), ((img_test_i[:,:,1]*256+128.0)).astype(np.uint8))
                    seg_output_idx_i = np.argmax(seg_output_i, axis=-1).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_seg_output.png".format(custom_status, i)), seg_output_idx_i*50)
                    img_vis = visualize_seg_detection(img_orig_i, seg_output_idx_i, energy_l_i, energy_r_i, box_gt_l = None, box_gt_r = None, close_kernel_size = close_kernel_size)
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_vis.png".format(custom_status, i)), img_vis.astype(np.uint8))
                    if config.energy:
                        energy_vis_l = (energy_l_i*255).astype(np.uint8)
                        _, energy_vis_l = cv2.threshold(energy_vis_l, 127, 255, cv2.THRESH_BINARY)
                        energy_vis_l = visualize_seg_detection(cv2.cvtColor(energy_vis_l, cv2.COLOR_GRAY2RGB), None, energy_l_i, None, box_gt_l = None, close_kernel_size = close_kernel_size)
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_energy_l.png".format(custom_status, i)), energy_vis_l)
                        energy_vis_r = (energy_r_i*255).astype(np.uint8)
                        _, energy_vis_r = cv2.threshold(energy_vis_r, 127, 255, cv2.THRESH_BINARY)
                        energy_vis_r = visualize_seg_detection(cv2.cvtColor(energy_vis_r, cv2.COLOR_GRAY2RGB), None, None, energy_r_i, box_gt_r = None, close_kernel_size = close_kernel_size)
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_energy_r.png".format(custom_status, i)), energy_vis_r)
            
    return iou_meter.avg, ap_meter.avg, inf_time_meter.avg