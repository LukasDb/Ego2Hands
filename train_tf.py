import numpy as np
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import AverageMeter


def train(config, model, adapt=False):
    print("Training for seg on Ego2Hands dataset.")

    # train
    hand_dataset_train = lambda : Ego2HandsData(config, mode = "train_seg", seq_i = -1)
    #elf.img_h, self.img_w = 
    temp = hand_dataset_train()
    h, w = (temp.img_h, temp.img_w) # 288, 512

    output_signature = (
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32),
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
    # if args.save_outputs:
        # adapt_status = "_custom" if args.custom else "_adapt_seq{}".format(seq_i) if args.adapt else ""
        # out_seg_path = "outputs/{}_{}_edge{}_energy{}_seg_train{}/".format(config.dataset, config.model_name, int(args.input_edge), int(args.energy), adapt_status)
        # if not os.path.exists(out_seg_path):
            # os.makedirs(out_seg_path)
    
    # For model save directory
    # Create save directory for model
    # model_dir_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
    # if not os.path.exists(model_dir_path):
        # os.makedirs(model_dir_path)
        
    # if not adapt:
        # model_save_path = os.path.join(model_dir_path, '{}_{}_seg'.format(config.dataset, config.model_name))
    # else:
        # model_save_path = os.path.join(model_dir_path, '{}_{}_seg_adapt_seq{}'.format(config.dataset, config.model_name, seq_i))

        
    # Criterions
    criterion_seg = tf.keras.losses.CategoricalCrossentropy(from_logits = True) #  nn.CrossEntropyLoss().cuda()
    criterion_mse = tf.keras.losses.MeanSquaredError() # nn.MSELoss().cuda()

    # Measures
    iou_val_best = 0.0
    loss_meters = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    
    # Training params
    if adapt:
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
    
    #optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr_rate, weight_decay=config.weight_decay)
    optimizer_seg = tf.keras.optimizers.Adam(learning_rate=lr_rate, beta_2=1-config.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_seg, step_size = step_size, gamma = gamma)

    # model.sigmoid.build()
    img_id_batch, img_orig_tensor, img_tensor, seg_tensor_, seg_1_2_tensor_, seg_1_4_tensor_, energy_gt_tensor, energy_gt_1_2_tensor, energy_gt_1_4_tensor = next(temp)

    model.build(input_shape = (config.batch_size, *img_tensor.shape))

    model.summary()

    while iters < max_iter:
        for i, (img_id_batch, img_orig_tensor, img_tensor, seg_tensor_, seg_1_2_tensor_, seg_1_4_tensor_, energy_gt_tensor, energy_gt_1_2_tensor, energy_gt_1_4_tensor) in enumerate(train_loader):
            iters += 1
            if iters > max_iter:
                break
            img_id = np.reshape(img_id_batch.numpy(), (-1))[0]
            img_batch_size = img_tensor.shape[0]
            img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

            # convert segmentation from shape (N,H,W,1) to (N,H,W,n_classes)
            seg_tensor = np.zeros((*seg_tensor_.shape[:3], config.num_classes))
            seg_1_2_tensor = np.zeros((*seg_1_2_tensor_.shape[:3], config.num_classes))
            seg_1_4_tensor = np.zeros((*seg_1_4_tensor_.shape[:3], config.num_classes))

            for i in range(config.num_classes):
                seg_tensor[seg_tensor_==i, i] = 1
                seg_1_2_tensor[seg_1_2_tensor_==i, i] = 1
                seg_1_4_tensor[seg_1_4_tensor_==i, i] = 1
            
            # Forward pass
            if "CSM" in config.model_name:
                if config.energy:
                    with tf.GradientTape() as tape:
                        seg_output1, energy_output1, seg_output_final, energy_output_final = model(img_tensor)
                        loss_seg1 = criterion_seg(seg_output1, seg_1_4_tensor)
                        loss_e1 = criterion_mse(energy_output1, energy_gt_1_4_tensor)
                        loss_seg2 = criterion_seg(seg_output_final, seg_1_2_tensor)
                        loss_e2 = criterion_mse(energy_output_final, energy_gt_1_2_tensor)
                        loss_seg_total = loss_seg1 + loss_e1 + loss_seg2 + loss_e2
                        
                    loss_meters[0].update(float(loss_seg1), img_batch_size)
                    loss_meters[1].update(float(loss_e1), img_batch_size)
                    loss_meters[2].update(float(loss_seg2), img_batch_size)
                    loss_meters[3].update(float(loss_e2), img_batch_size)
                else:
                    with tf.GradientTape() as tape:
                        seg_output1, seg_output_final = model(img_tensor)
                        loss_seg1 = criterion_seg(seg_output1, seg_1_4_tensor)
                        loss_seg2 = criterion_seg(seg_output_final, seg_1_2_tensor)
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
                print('Train Iteration: {} iters'.format(iters))
                print('Loss_seg_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[0]))
                print('Loss_energy_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[1]))
                print('Loss_seg_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[2]))
                print('Loss_energy_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[3]))

                #iou_np = compute_iou(seg_output_final, seg_gt_var)
                #print("IoU sample = {}".format(iou_np))

                # Visualize Outputs
                if config.save_outputs:
                    img_orig_np = img_orig_tensor.cpu().data.numpy()
                    img_np = img_tensor.cpu().data.numpy().transpose(0,2,3,1)
                    seg_output_np = seg_output_final.cpu().data.numpy().transpose(0,2,3,1)
                    seg_gt_np = seg_gt_var.cpu().data.numpy()
                    energy_output_np = energy_output_final.cpu().data.numpy().transpose(0,2,3,1)
                    energy_gt_np = energy_gt_var.cpu().data.numpy().transpose(0,2,3,1)

                    for batch_i, (img_orig_i, img_i, seg_output_i, seg_gt_i, energy_output_i, energy_gt_i) in enumerate(zip(img_orig_np, img_np, seg_output_np, seg_gt_np, energy_output_np, energy_gt_np)):
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_orig.png".format(iters, batch_i)), (img_orig_i).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_grayscale.png".format(iters, batch_i)), (img_i[:,:,0]*256.0 + 128.0).astype(np.uint8))
                        if args.input_edge:
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_edge.png".format(iters, batch_i)), (img_i[:,:,1]*256.0 + 128.0).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_output.png".format(iters, batch_i)), np.argmax(seg_output_i, axis=-1).astype(np.uint8)*50)
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_gt.png".format(iters, batch_i)), seg_gt_i*50)
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,1]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,2]*255).astype(np.uint8))
                        #cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,0]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_output.png".format(iters, batch_i)), (energy_output_i[:,:,1]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_output.png".format(iters, batch_i)), (energy_output_i[:,:,2]*255).astype(np.uint8))
                        #cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_output.png".format(iters, batch_i)), (energy_output_i[:,:,0]*255).astype(np.uint8))
                        
                # Clear meters
                for loss_meter in loss_meters:
                    loss_meter.reset()
            
            # Save models
            if iters % config.save_interval == 0:
                if not args.custom:
                    model_is_best = False                  
                    if not args.adapt:
                        iou_meter_val = AverageMeter()
                        for seq_j in range(0, config.num_seqs):
                            seq_j = seq_j + 1
                            iou_seq_j, ap_seq_j, inf_time_j = test_ego2hands_seg(model_seg, seq_j, args, config, energy_status = energy_status)
                            print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_j, iou_seq_j, ap_seq_j, inf_time_j))
                            model_seg.train()
                            if iou_seq_j >= 0:
                                iou_meter_val.update(iou_seq_j, 1)
                        print("Mean eval iou = {}".format(iou_meter_val.avg))
                        if iou_meter_val.avg >= iou_val_best:
                            iou_val_best = iou_meter_val.avg
                            model_is_best = True
                            print("New best IoU set")  
                    else:
                        iou_seq_j, ap_seq_j, inf_time_j = test_ego2hands_seg(model_seg, seq_i, args, config, energy_status = energy_status)
                        if iou_seq_j >= iou_val_best:
                            iou_val_best = iou_seq_j
                            model_is_best = True
                        print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_i, iou_seq_j, ap_seq_j, inf_time_j))
                        model_seg.train()
                    
                    if model_is_best:
                        print("Saving best model at {}".format(model_save_path))
                        save_model({
                             'iter': iters,
                             'state_dict': model_seg.state_dict(),
                        }, is_best = True, is_last = False, filename = model_save_path)
                    
                print("Saving latest model at {}".format(model_save_path))
                save_model({
                     'iter': iters,
                     'state_dict': model_seg.state_dict(),
                }, is_best = False, is_last = False, filename = model_save_path)
                
    # Save the last model as pretrained model
    save_model({
         'iter': iters,
         'state_dict': model_seg.state_dict(),
    }, is_best = False, is_last = True, filename = model_save_path)