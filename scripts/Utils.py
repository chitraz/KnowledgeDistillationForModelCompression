# re-implementation of vanilla KD  
from KD_methods import KD_Hinton
from KD_methods import DISTV2

# other KD methods taken from the distiller zoo
from distiller_zoo import Attention
from distiller_zoo import DIST

# feature adaptors 
from distiller_zoo.AIN import transfer_conv # for SRD
from models.util import ConvReg # for FitNet

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import copy
import time
import os
import sys
import datetime
from tqdm import tqdm


'''##########################################################################################################################
        Evaluate model 
##########################################################################################################################'''
def evaluate(model, device, Dataloader_test, NUM_CLASS = 10, ConfMatrix = False):
    total_samples = len(Dataloader_test.dataset)
    total_correct = 0
    acc_top1 = 0.0

    with torch.no_grad():
        model.to(device)
        model.eval()

        if ConfMatrix == True:
            # Tensors to store the confusion matrix and the per-class precision and recall accuracies computed from it
            ConfusionMatrix = torch.zeros([NUM_CLASS, NUM_CLASS], dtype=torch.int32, device=device)
            precision = torch.zeros([NUM_CLASS], dtype=torch.float)
            recall = torch.zeros([NUM_CLASS], dtype=torch.float)

        # Iterate test batches
        for _ , batch in tqdm(enumerate(Dataloader_test), total = len(Dataloader_test)):
            Images, GTs, _ = batch
            Images = Images.to(device)
            GTs = GTs.to(device)
            GTs = GTs.view(-1)
            Preds = model(Images)
            Preds = Preds.argmax(1)

            # Accumulate number of correct predictions
            total_correct += torch.sum(Preds==GTs)
            
            # Populate a Confusion Matrix, accumulated across all batches 
            if ConfMatrix == True:
                for GT_Class in range(NUM_CLASS):
                    Mask1 = (GTs == GT_Class)
                    for Pred_Class in range(NUM_CLASS):
                        Mask2 = (Preds == Pred_Class)
                        ConfusionMatrix[Pred_Class, GT_Class] += torch.sum(Mask1*Mask2, dtype=torch.int32)

    # Compute ACC@1 
    acc_top1 = 100 * total_correct.item()/total_samples

    # Compute per-class precision and recall accuracies from Confusion Matrix  
    if ConfMatrix == True:
        CM = ConfusionMatrix
        for i in range(NUM_CLASS):
            precision[i] = 100 * CM[i,i]/float(CM.sum(1)[i])
            recall[i] = 100 * CM[i,i]/float(CM.sum(0)[i])
        return acc_top1, ConfusionMatrix, precision, recall
    else:
        return acc_top1


'''##########################################################################################################################
        Functions to pre-compute, and fetch, teacher's ouput logits
##########################################################################################################################'''
def PreCompute_logits(model, device, Dataloader_train):
    MAP_ImgIdx_logits = {}
    with torch.no_grad():
        model.to(device)
        model.eval()
        print('Pre-computing teacher oputput logts for all train samples...')
        for _ , batch in tqdm(enumerate(Dataloader_train), total = len(Dataloader_train)):    
            img_batch, _ , idxs = batch # unpack        
            img_batch = img_batch.to(device) # put images on gpu mem
            pred_logits = model(img_batch) # forward pass to get predicted logits 

            # save mapping as a Dict: image index -> output logits 
            for (idx, logits) in zip(idxs,pred_logits): 
                MAP_ImgIdx_logits[idx.item()] = logits

    return MAP_ImgIdx_logits

def fetch_logtis(teacher_map, idxs):
    # get number of samples in batch 
    batch_size = len(idxs)
    # get number of classes by checking size of logit vector
    num_class = len(list(teacher_map.values())[0])
    
    # construct tensor of correct size to hold the fetched logits
    logits_batch = torch.zeros((batch_size, num_class), dtype=torch.float32)
    count = 0 
    
    # fetch the logits
    for idx in idxs:
        logits_batch[count, :] = teacher_map[idx.item()]
        count += 1

    return logits_batch


'''##########################################################################################################################
        main training loop
##########################################################################################################################'''
def train(model, teacher, device, args, Dataloader_train, Dataloader_valid, writer):

    # create name of current training run 
    if args.mode == 'from-scratch':
        run_name = args.student + '_'  + args.mode + '_' + args.dataset 
    else:
        run_name = args.student + '_'  + args.mode + '_'  + args.teacher + '_' + args.dataset 

    # Cross-entroy loss function for the standard classification task loss
    CE_loss = torch.nn.CrossEntropyLoss() 

    # load to be trained model (student) onto GPU
    model.to(device)

    # list of modules to be trained (student + relavent feature adaptors)
    train_mod = torch.nn.ModuleList([])

    # add to-be trained model (student) to train list 
    train_mod.append(model)

    # if running KD method, prepare the teacher model, relavent loss functions and connectors
    if args.mode != 'from-scratch': 
        
        # put pre-trained teacher model on GPU mem and set to eval mode
        teacher.to(device)
        teacher.eval()  # should set to eval or train mode? how is teacher BN layers handled? 

        # feedfoward a random train batch to determine the shapes of teacher's and student's activation tensors
        dummy_input, _, _ = next(iter(Dataloader_train))
        dummy_input = dummy_input.to(device)
        print('Checking student and teacher feature sizes...')
        with torch.no_grad():
            Sfeats, Slogits = model(dummy_input, is_feat=True, preact=False)   
            Tfeats, Tlogits = teacher(dummy_input, is_feat=True, preact=False)
            
        # print feature shapes
        print('Student shapes:')
        for f in Sfeats:
            print(f.shape)
        print('Teacher shapes:')
        for f in Tfeats:
            print(f.shape)
            
        # pre-compute teacher's ouput logits for all images in training set if option selected
        if args.pre_comp:
            teacher_map = PreCompute_logits(teacher, device, Dataloader_train)

        # prepare the chosen KD method
        if args.mode == 'KD_Hinton': # vanilla KD 
            Distil_loss = KD_Hinton(args.T) 

        elif args.mode == 'FitNet': # FitNet Stage-2 (finish training with vanilla KD)
            Distil_loss = KD_Hinton(args.T) 
            
        elif args.mode == 'FitNet-like': # FitNet but in single stage. (Hint based loss + KD loss)
            MSE_loss = torch.nn.MSELoss()
            Distil_loss = KD_Hinton(args.T)
            
            hint_l = args.hint_layer # hint layer 
            guide_l = args.hint_layer # guided layer
            
            # find teacher and student shape
            dummy_input, _, _ = next(iter(Dataloader_train))
            dummy_input = dummy_input.to(device)
            with torch.no_grad():
                Sfeats, Slogits = model(dummy_input, is_feat=True, preact=False)   
                Tfeats, Tlogits = teacher(dummy_input, is_feat=True, preact=False)
            
            # convolutional regressor 
            adaptor = ConvReg(Sfeats[guide_l].shape,Tfeats[hint_l].shape)
            adaptor.to(device)
            train_mod.append(adaptor) 
            
        elif args.mode == 'AT': # Attention Transfer 
            
            # MSE loss between teacher's and student's attention maps
            AT_loss = Attention()  
            
            # also option for applying vanilla KD 
            Distil_loss = KD_Hinton(args.T) 
            
            # 1x1 conv (+ BN + Relu) feature adaptor to transform student's feature to match teacher's channel dim
            #adaptor = transfer_conv(Sfeats[-2].shape[1],Tfeats[-2].shape[1])
            #adaptor.to(device)
            #dtrain_mod.append(adaptor) # also train adaptor 
        
        elif args.mode == 'DML': # Deep Mutual Learning
            
            # create an optimzer (same settings as student's optimzer) for teacher network
            optimizer_T = optim.SGD(teacher.parameters(), lr=args.BASE_LR, momentum=args.MOMENTUM, weight_decay=args.W_DECAY)
            lr_policy_T = lr_scheduler.MultiStepLR(optimizer_T, args.STEP_DOWNS, gamma=args.STEP_GAMMA)
            
            # DML loss, same as vanilla KD loss but with T=1 (not quite? also no T^2 scaling term)
            DML_loss = KD_Hinton(T=1) 
                
        elif args.mode == 'DIST': # 'Knowledge Distillation from A Stronger Teacher'
            # BETA*(inter-class loss) + GAMMA*(intra-class loss)
            DIST_loss = DIST(args.BETA, args.GAMMA, args.T) 
            # BETA: weight balance for L_inter
            # GAMMA: weight balance for L_intra
            # T: softmax temperture            
            
        elif args.mode in ['SRD', 'SRDwithPMSE', 'SRDwithKL', 'SRDwithDIST']: # SRD with various loss designs ('SRD' is the defualt with mse)
              
            if args.mode == 'SRDwithKL':
                KL_loss = KD_Hinton(T=1) # KL div
                
            if args.mode == 'SRDwithDIST':
                # relaxed logit matching loss from DIST: L_inter + L_intra 
                DIST_loss = DISTV2(beta=1,gamma=1) # L_inter and L_intra equally balanced
            
            # Mean Squared Error loss for direct intermediate feature matching and also for SRD loss if using 'SRD' and 'SRDwithPMSE'
            MSE_loss = torch.nn.MSELoss() 

            # 1x1 conv (+ BN + Relu) feature adaptor to transform student's feature to match teacher's channel dim
            # can only handle mismatch in #channels between T and S repersentations
            adaptor = transfer_conv(Sfeats[-2].shape[1],Tfeats[-2].shape[1]) 
            adaptor.to(device)
            train_mod.append(adaptor) # add adaptor to train list 
            
            # prepare pre-trained teacher's classifier (const)
            teacher_classifier = torch.nn.Sequential(
                torch.nn.AvgPool2d(Tfeats[-2].size(2)), # average pool: [NxCx8x8]->[NxCx1x1]
                torch.nn.Flatten(), # flatten: [NxCx1x1] -> [NxC]
                teacher.fc # teacher pre-trained fc layer
                ) 
            teacher_classifier.to(device)
            #freeze classifier weights
            for param in teacher_classifier.parameters():
                param.requires_grad = False

    # how often to print the training loss (in terms of # training iterations)
    print_freq = 100

    # setup optimiser and learning rate scheduler 
    optimizer = optim.SGD(train_mod.parameters(), lr=args.BASE_LR, momentum=args.MOMENTUM, weight_decay=args.W_DECAY)
    lr_policy = lr_scheduler.MultiStepLR(optimizer, args.STEP_DOWNS, gamma=args.STEP_GAMMA)

    # run training loop!
    Best_ValAcc = 0.0
    for epoch in tqdm(range(args.MAX_EPOCH), total=args.MAX_EPOCH, desc='epochs'): # iterate epochs
        total_trainloss = 0.0
        total_trainloss_T = 0.0 # used only for DML for monitoring teacher's training loss

        for itr, Batch in enumerate(Dataloader_train): # iterate train mini-batches

            # unpack batch and load into GPU
            Images, GTs, idxs = Batch
            Images = Images.to(device)
            GTs = GTs.to(device)
            
            # set model in training mode i.e. with drop out, BN layers active etc
            model.train()
            
            # clear all previous gradients in model parameters
            optimizer.zero_grad()
            
            # forward pass
            feats, Preds = model(Images, is_feat=True, preact=False) # feature tensors after Relu 

            # reshape gt batch
            GTs = GTs.view(-1) # [batchsize x 1] -> [batchsize]

            # compute training loss 
            if args.mode == 'from-scratch': # vanilla training, no teacher supervision
                # standard classification loss
                train_loss = CE_loss(Preds, GTs) 
            
            elif args.mode == 'DML': # Deep Mutual Learning
                # set teacher in train mode
                teacher.train()
                # zero out all previous gradient in teacher's parameters
                optimizer_T.zero_grad()
                # forward pass teacher network 
                T_feats, T_Preds = teacher(Images, is_feat=True, preact=False)
                
                # compute teacher's loss 
                train_loss_T = CE_loss(T_Preds, GTs) + args.ALPHA * DML_loss(T_Preds, Preds.detach())
                
                # compute student's loss 
                train_loss = CE_loss(Preds, GTs) + args.BETA * DML_loss(Preds, T_Preds.detach())
                
                # update teacher 
                train_loss_T.backward()
                optimizer_T.step() 
                
                # accumulate teacher's training loss  
                total_trainloss_T += train_loss_T.item()
                
            else: # Offline KD methods
                
                # pre-compute teacher's logits if option selected
                if args.pre_comp:
                    T_Preds = fetch_logtis(teacher_map, idxs).to(device)
                else:     
                    # forward pass teacher 
                    with torch.no_grad():
                        T_feats, T_Preds = teacher(Images, is_feat=True, preact=False)

                # compute the relavent student loss functions for the KD method chosen
                if args.mode == 'KD_Hinton':
                    # compute loss: classification loss + KD loss
                    train_loss = (1.0-args.ALPHA)*CE_loss(Preds, GTs) +  args.ALPHA*Distil_loss(Preds, T_Preds)
                
                elif args.mode == 'FitNet':
                    # Stage-2: finish training using vanilla KD
                    # compute loss: classification loss + KD loss
                    train_loss = (1.0-args.ALPHA)*CE_loss(Preds, GTs) +  args.ALPHA*Distil_loss(Preds, T_Preds)
                    
                elif args.mode == 'FitNet-like': # FitNet but done in a single stage. 
                    
                    # adapt student feature
                    feats_adapted = adaptor(feats[guide_l])
                    
                    # Hint based loss: regression 
                    hint_loss = MSE_loss(feats_adapted, T_feats[hint_l]) 
                    
                    # training loss func: (1 - A)*task_loss + A*KD_loss + B*Hint_loss
                    train_loss = (1.0-args.ALPHA)*CE_loss(Preds, GTs) +  args.ALPHA*Distil_loss(Preds, T_Preds) + args.BETA*hint_loss
                
                elif args.mode == 'AT':
                    # TODO:need to add adaptor at each AT point! 
                    
                    # Attention loss after each layer groups and also after avg_pool. [NxCx32x32], [NxCx16x16], [NxCx8x8], [NxC]
                    AT_losses = AT_loss(feats[1:-1], T_feats[1:-1])   
                    AT_losses = sum(AT_losses) 

                    # construct training loss: (1-alpha)*CE_loss + alpha*KD_Hinton_loss + beta*Attention_loss 
                    train_loss = (1.0-args.ALPHA)*CE_loss(Preds, GTs) +  args.ALPHA*Distil_loss(Preds, T_Preds) + args.BETA* AT_losses
                
                elif args.mode == 'DIST': 
                    # compute loss: classification loss + (inter_class loss + intra_class loss)          
                    train_loss = args.ALPHA*CE_loss(Preds, GTs) + DIST_loss(Preds, T_Preds)
                
                elif args.mode in ['SRD', 'SRDwithPMSE', 'SRDwithKL', 'SRDwithDIST']: # SRD method with different SRD loss designs: mse(), kl() and DIST(). 
                    # transfrom student feature 
                    feats_adapted = adaptor(feats[-2])  
                    
                    # pass the student's (adapted) feature through teacher's pre-trained classifier to get the cross-network logits
                    S_cross_logits = teacher_classifier(feats_adapted)
            
                    if args.mode == 'SRD': # (defualt) MSE() for SRD loss
                        SRD_loss = MSE_loss(S_cross_logits, T_Preds)
                        
                    elif args.mode == 'SRDwithPMSE': # MSE() after softmax normalization
                        # apply softmax 
                        S_cross_prob = S_cross_logits.softmax(dim=1)
                        teacher_prob = T_Preds.softmax(dim=1)
                        # SRD loss using mse
                        SRD_loss = MSE_loss(S_cross_prob, teacher_prob)
                        
                    elif args.mode == 'SRDwithKL': # KL div after softmax normalization
                        # apply softmax 
                        S_cross_prob = S_cross_logits.softmax(dim=1)
                        teacher_prob = T_Preds.softmax(dim=1)
                        # SRD loss using KL div
                        SRD_loss = KL_loss(S_cross_prob, teacher_prob)
                        
                    elif args.mode == 'SRDwithDIST': # relaxed matching from DIST method. (apply directly on logits, no softmax norm) 
                        # SRD loss using L_srd = L_inter + L_intra (equally blanced)
                        
                        SRD_loss = DIST_loss(S_cross_logits, T_Preds) 
                    
                    # # Feature Matching loss   
                    FM_loss = MSE_loss(feats_adapted, T_feats[-2]) 

                    # training loss = (task loss) + alpha*(FM loss) + beta*(SRD loss) 
                    train_loss = CE_loss(Preds, GTs) + args.ALPHA*FM_loss + args.BETA*SRD_loss                              
                
            # backward pass
            train_loss.backward()
            
            # update step
            optimizer.step()
            
            # accumulate training batch loss
            total_trainloss += train_loss.item()
            
            # log training loss, lr value and train acc @1 every print_freq iterations
            if ((itr+1) % print_freq)==0:
                avg_TrainBatchLoss = total_trainloss / (itr+1)
                writer.add_scalar('Loss/train', avg_TrainBatchLoss, (epoch*len(Dataloader_train) + itr))
                current_lr = lr_policy.get_last_lr()[0]
                writer.add_scalar('LR/Learning rate policy', current_lr, (epoch*len(Dataloader_train) + itr))
                
                # also log teacher's training if running DML, since teacher is also trained
                if args.mode == 'DML':
                    avg_TrainBatchLoss_T = total_trainloss_T / (itr+1)
                    writer.add_scalar('Loss/train(teacher)', avg_TrainBatchLoss_T, (epoch*len(Dataloader_train) + itr))
                    current_lr = lr_policy_T.get_last_lr()[0]
                    writer.add_scalar('LR/Learning rate policy(teacher)', current_lr, (epoch*len(Dataloader_train) + itr))

        # tick the lr scheduler every epoch 
        lr_policy.step()
        # also tick lr scheduler for teacher training if running DML 
        if args.mode == 'DML':
            lr_policy_T.step()

        # evaluate model on valid set after every training epoch
        with torch.no_grad(): 
            total_validloss = 0.0
            total_validcorrect = 0
            total_validloss_T = 0.0 # for teacher if running DML 
            total_validcorrect_T = 0 # for teacher if running DML
            
            for _, valid_Batch in enumerate(Dataloader_valid): # iterate valid mini-batches
                # unpack batch and load into GPU device 
                valid_Images, valid_GTs, _ = valid_Batch
                valid_Images = valid_Images.to(device)
                valid_GTs = valid_GTs.to(device)

                # set model in test mode i.e. with drop out disabled
                model.eval()

                # forward pass to get our model predictions
                valid_Preds = model(valid_Images)

                # [batchsize x 1] -> [batchsize]
                valid_GTs = valid_GTs.view(-1) 

                # compute validation loss
                # **note: only monitoring the classification loss
                valid_loss = CE_loss(valid_Preds, valid_GTs) 

                # accumulate batch loss
                total_validloss += valid_loss.item()

                # get predicted class integer for each sample in batch
                valid_Preds = valid_Preds.argmax(1)
            
                # accumulate the number of correct predicitions 
                total_validcorrect += torch.sum(valid_Preds==valid_GTs)
                
                # also monitor teacher if it's being trained
                if args.mode == 'DML':
                    teacher.eval()
                    valid_Preds_T = teacher(valid_Images)
                    valid_loss_T = CE_loss(valid_Preds_T, valid_GTs)
                    total_validloss_T += valid_loss_T.item()
                    valid_Preds_T = valid_Preds_T.argmax(1)
                    total_validcorrect_T += torch.sum(valid_Preds_T==valid_GTs) 

            # get average validation batch loss
            avg_ValidBatchLoss = total_validloss / float(len(Dataloader_valid))
            writer.add_scalar('Loss/valid', avg_ValidBatchLoss, epoch*len(Dataloader_train))
            # get Top-1 validation acc 
            ValidAcc = total_validcorrect.item() / float(len(Dataloader_valid.dataset)) * 100.0
            writer.add_scalar('Accuracy/Top1_valid', ValidAcc, epoch*len(Dataloader_train))
            
            # also log teacher training if it's being trained 
            if args.mode == 'DML':
                avg_ValidBatchLoss_T = total_validloss_T / float(len(Dataloader_valid))
                writer.add_scalar('Loss/valid(teacher)', avg_ValidBatchLoss_T, epoch*len(Dataloader_train))
                ValidAcc_T = total_validcorrect_T.item() / float(len(Dataloader_valid.dataset)) * 100.0
                writer.add_scalar('Accuracy/Top1_valid(teacher)', ValidAcc_T, epoch*len(Dataloader_train))
            
        # model checkpoint every epoch
        time_now = time.time()
        datestamp = str(datetime.date.fromtimestamp(time_now)) 
        file_name = run_name + '_' + args.Exp_Name + '_' + str(epoch + 1) +  'epochs_ValAcc:' + str(round(ValidAcc, 2)) + '_'+ datestamp + '.pth'
        save_path = os.path.join('snaps', args.Exp_Name, run_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_location = os.path.join(save_path, file_name)
        torch.save(model.state_dict(),save_location) 


'''##########################################################################################################################
        initialization training 
##########################################################################################################################'''
def init_train(model, teacher, device, args, Dataloader_train, Dataloader_valid, writer):  
    
    if args.mode == 'FitNet': # Hint based training for FitNet
        
        # list of modules to be trained (student + relavent feature adaptors)
        train_mod = torch.nn.ModuleList([])
        model.to(device)
        train_mod.append(model)
        teacher.to(device)
        teacher.eval()

        MSE_loss = torch.nn.MSELoss()
        
        hint_l = args.hint_layer # hint layer 
        guide_l = args.hint_layer # guided layer
        
        # find teacher and student shape
        dummy_input, _, _ = next(iter(Dataloader_train))
        dummy_input = dummy_input.to(device)
        with torch.no_grad():
            Sfeats, Slogits = model(dummy_input, is_feat=True, preact=False)   
            Tfeats, Tlogits = teacher(dummy_input, is_feat=True, preact=False)
        
        # convolutional regressor as feature adaptor
        adaptor = ConvReg(Sfeats[guide_l].shape,Tfeats[hint_l].shape)
        adaptor.to(device)
        train_mod.append(adaptor) 
        
        # setup optimiser and learning rate scheduler for stage 1 training 
        S1_MAX_Epoch = 50
        optimizer = optim.SGD(train_mod.parameters(), lr=args.BASE_LR, momentum=args.MOMENTUM, weight_decay=args.W_DECAY)
        lr_policy = lr_scheduler.MultiStepLR(optimizer, [20, 35], gamma=args.STEP_GAMMA)
        
        print_freq = 100
        # run Stage-1 training!
        for epoch in tqdm(range(S1_MAX_Epoch), total=S1_MAX_Epoch, desc='epochs'): # iterate epochs
            total_trainloss = 0.0      
            for itr, Batch in enumerate(Dataloader_train): # iterate train mini-batches
                Images, _ , _ = Batch  
                Images = Images.to(device)
                model.train()
                optimizer.zero_grad()
                
                # forward pass student
                feats, _ = model(Images, is_feat=True, preact=False) # feature tensors after Relu 
                
                # forward pass teacher
                with torch.no_grad():
                    T_feats, T_Preds = teacher(Images, is_feat=True, preact=False)
                    
                # adapt student feature
                feats_adapted = adaptor(feats[guide_l]) 

                # Hint based loss: regression 
                train_loss = MSE_loss(feats_adapted, T_feats[hint_l]) 
                
                # backward pass
                train_loss.backward()
                # update step
                optimizer.step()
                # accumulate training batch loss
                total_trainloss += train_loss.item()
                # log training loss, lr value and train acc @1 every print_freq iterations
                if ((itr+1) % print_freq)==0:
                    avg_TrainBatchLoss = total_trainloss / (itr+1)
                    writer.add_scalar('Loss/train (stage-1)', avg_TrainBatchLoss, (epoch*len(Dataloader_train) + itr))
                    current_lr = lr_policy.get_last_lr()[0]
                    writer.add_scalar('LR/Learning rate policy (stage-1)', current_lr, (epoch*len(Dataloader_train) + itr))
            # tick the lr scheduler every epoch 
            lr_policy.step()
