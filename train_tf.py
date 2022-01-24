import numpy as np
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import AverageMeter, save_model
import cv2
from models.CSM.CSM_tf import CSM_baseline
import cv2
from data_loaders.Ego2Hands_tf import Ego2HandsData
from test_tf import test_tf
import os
from utils import Config
import numpy as np

def main():
    config_path = os.path.join("configs", "config_tf.yml")
    config = Config(config_path)
    model = CSM_baseline(n_classes = config.num_classes, with_energy = config.energy, input_edge = config.input_edge)
    train(config, model)


def train(config, model):
    print("Training for seg on Ego2Hands dataset.")

    seq_i = -1

    # train
    hand_dataset_train = lambda : Ego2HandsData(config, mode = "train_seg", seq_i = seq_i)
    #elf.img_h, self.img_w = 
    temp = hand_dataset_train()
    h, w = (temp.img_h, temp.img_w) # 288, 512
    
    
    #output:  [img_id_batch, img_orig_tensor, img_tensor, 
    #           seg_tensor, seg_1_2_tensor, seg_1_4_tensor, 
    #           energy_gt_tensor, energy_gt_1_2_tensor, energy_gt_1_4_tensor)]
    output_signature = (
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 2 if config.energy else 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w), dtype=tf.uint8),
        tf.TensorSpec(shape=(h//2, w//2), dtype=tf.uint8),
        tf.TensorSpec(shape=(h//4, w//4), dtype=tf.uint8),
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h//2, w//2, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h//4, w//4, 3), dtype=tf.float32),
    )
    #[print(x.dtype) for x in next(temp)]
    #[print(x.shape) for x in next(temp)]

    train_loader = tf.data.Dataset.from_generator(hand_dataset_train, output_signature=output_signature) \
        .cache().batch(config.batch_size).shuffle(20).prefetch(tf.data.AUTOTUNE)

    print("Dataset loaded. #instances = {}".format(temp.__len__()))
        
    # For output directory
    if config.save_outputs:
        adapt_status = "_adapt_seq{}".format(seq_i) if config.adapt else ""
        out_seg_path = "outputs/{}_{}_edge{}_energy{}_seg_train{}/".format(config.dataset, config.model_name, int(config.input_edge), int(config.energy), adapt_status)
        if not os.path.exists(out_seg_path):
            os.makedirs(out_seg_path)
    
    # For model save directory
    # Create save directory for model
    energy_status = "with_energy" if config.energy else "without_energy"
    model_dir_path = os.path.join(config.model_path, config.dataset, config.model_name, energy_status)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
      
    if not config.adapt:
        model_save_path = os.path.join(model_dir_path, '{}_{}_seg'.format(config.dataset, config.model_name))
    else:
        model_save_path = os.path.join(model_dir_path, '{}_{}_seg_adapt_seq{}'.format(config.dataset, config.model_name, seq_i))

        
    # Criterions
    criterion_seg = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    criterion_mse = tf.keras.losses.MeanSquaredError()

    # Measures
    iou_val_best = 0.0
    loss_meters = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    
    # Training params
    if config.adapt:
        lr_rate = config.base_lr_seg_adapt
        step_size = config.policy_parameter_seg_adapt.step_size 
        gamma = config.policy_parameter_seg_adapt.gamma
        iters = 0
        max_iter = config.max_iter_seg_adapt
    else:
        lr_rate = config.base_lr_seg
        step_size = config.policy_parameter_seg.step_size 
        gamma = config.policy_parameter_seg.gamma
        iters = 0
        max_iter = config.max_iter_seg
    
    
    optimizer_seg = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    


    _, _, img_tensor, _, _, _, _, _, _ = next(temp)

    model.build(input_shape = (config.batch_size, *img_tensor.shape))

    model.summary()

    while iters < max_iter:
        for i, (img_id_batch, img_orig_tensor, img_tensor, seg_tensor, seg_1_2_tensor, seg_1_4_tensor, energy_gt_tensor, energy_gt_1_2_tensor, energy_gt_1_4_tensor) in enumerate(train_loader):
            iters += 1
            if iters > max_iter:
                break
            img_id = np.reshape(img_id_batch.numpy(), (-1))[0]
            img_batch_size = img_tensor.shape[0]
            img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
            
            # Forward pass
            if "CSM" in config.model_name:
                if config.energy:
                    with tf.GradientTape() as tape:
                        seg_output1, energy_output1, seg_output_final, energy_output_final = model(img_tensor)
                        loss_seg1 = criterion_seg(seg_1_4_tensor, seg_output1)
                        loss_e1 = criterion_mse(energy_gt_1_4_tensor, energy_output1)
                        loss_seg2 = criterion_seg(seg_1_2_tensor, seg_output_final)
                        loss_e2 = criterion_mse(energy_gt_1_2_tensor, energy_output_final)
                        loss_seg_total = loss_seg1 + loss_e1 + loss_seg2 + loss_e2
                        
                    loss_meters[0].update(float(loss_seg1), img_batch_size)
                    loss_meters[1].update(float(loss_e1), img_batch_size)
                    loss_meters[2].update(float(loss_seg2), img_batch_size)
                    loss_meters[3].update(float(loss_e2), img_batch_size)
                else:
                    with tf.GradientTape() as tape:
                        seg_output1, seg_output_final = model(img_tensor)
                        loss_seg1 = criterion_seg(seg_1_4_tensor, seg_output1)
                        loss_seg2 = criterion_seg(seg_1_2_tensor, seg_output_final)
                        loss_seg_total = loss_seg1 + loss_seg2
                
                    loss_meters[0].update(float(loss_seg1), img_batch_size)
                    loss_meters[1].update(float(0.0), img_batch_size)
                    loss_meters[2].update(float(loss_seg2), img_batch_size)
                    loss_meters[3].update(float(0.0), img_batch_size)
                    
                    energy_output_final = tf.zeros((img_batch_size, 3, 1, 1))
            
            grads = tape.gradient(loss_seg_total, model.trainable_weights)
            optimizer_seg.apply_gradients(zip(grads, model.trainable_weights))

            # Display seg info
            if iters % config.display_interval == 0:
                print('\n--- Train Iteration: {} iters ---'.format(iters))
                print('Loss_seg_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[0]))
                print('Loss_energy_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[1]))
                print('Loss_seg_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[2]))
                print('Loss_energy_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[3]))

                #iou_np = compute_iou(seg_output_final, seg_gt_var)
                #print("IoU sample = {}".format(iou_np))

                # Visualize Outputs
                if config.save_outputs:
                    img_orig_np = img_orig_tensor.numpy()
                    img_np = img_tensor.numpy()
                    seg_output_np = seg_output_final.numpy()
                    seg_gt_np = seg_tensor.numpy()
                    energy_output_np = energy_output_final.numpy()
                    energy_gt_np = energy_gt_tensor.numpy()

                    for batch_i, (img_orig_i, img_i, seg_output_i, seg_gt_i, energy_output_i, energy_gt_i) in enumerate(zip(img_orig_np, img_np, seg_output_np, seg_gt_np, energy_output_np, energy_gt_np)):
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_orig.png".format(iters, batch_i)), (img_orig_i).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_grayscale.png".format(iters, batch_i)), (img_i[:,:,0]*256.0 + 128.0).astype(np.uint8))
                        if config.input_edge:
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_edge.png".format(iters, batch_i)), (img_i[:,:,1]*256.0 + 128.0).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_output.png".format(iters, batch_i)), np.argmax(seg_output_i, axis=-1).astype(np.uint8)*100)
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_gt.png".format(iters, batch_i)), seg_gt_i*50)
                        if config.energy:
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,1]*255).astype(np.uint8))
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,2]*255).astype(np.uint8))
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,0]*255).astype(np.uint8))
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_output.png".format(iters, batch_i)), (energy_output_i[:,:,1]*255).astype(np.uint8))
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_output.png".format(iters, batch_i)), (energy_output_i[:,:,2]*255).astype(np.uint8))
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_output.png".format(iters, batch_i)), (energy_output_i[:,:,0]*255).astype(np.uint8))
                        
                # Clear meters
                for loss_meter in loss_meters:
                    loss_meter.reset()
            
            # Save models
            if iters % config.save_interval == 0:
                model_is_best = False                  
                if not config.adapt:
                    iou_meter_val = AverageMeter()
                    for seq_j in range(0, config.num_seqs):
                        seq_j = seq_j + 1
                        iou_seq_j, ap_seq_j, inf_time_j = test_tf(model, config, seq_j)
                        print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_j, iou_seq_j, ap_seq_j, inf_time_j))
                        if iou_seq_j >= 0:
                            iou_meter_val.update(iou_seq_j, 1)
                    print("Mean eval iou = {}".format(iou_meter_val.avg))
                    if iou_meter_val.avg >= iou_val_best:
                        iou_val_best = iou_meter_val.avg
                        model_is_best = True
                        print("New best IoU set")  
                else:
                    iou_seq_j, ap_seq_j, inf_time_j = test_tf(model, config)
                    if iou_seq_j >= iou_val_best:
                        iou_val_best = iou_seq_j
                        model_is_best = True
                    print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_i, iou_seq_j, ap_seq_j, inf_time_j))
                
                if model_is_best:
                    print("Saving best model at {}".format(model_save_path))
                    save_model(model, True, False, model_save_path)

            # after one step
            #print("Saving latest model at {}".format(model_save_path))
            #save_model(model, False, False, model_save_path)

    # after training
    save_model(model, False, True, model_save_path)


if __name__=='__main__':
    main()