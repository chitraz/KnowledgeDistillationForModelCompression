from Dataset import get_cifar10, get_cifar100, ImageNetDataset
from Models import get_Model
from Utils import init_train, train, evaluate

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
torch.backends.cudnn.benchmark = True 
from torchinfo import summary

# worker thread crash issue fix when using ImageNet. change defualt sharing strategy.
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import sys
import os
import json

if __name__ == '__main__':

    '''##########################################################################################################################
            Training Setup and Hyper-parameters 
    ##########################################################################################################################'''
    # parse the training setup and hyper-parameters from the command-line argument
    # namespace(dataset,mode,student,teacher,weights, .. )
    parser = argparse.ArgumentParser(prog='Distiller')

    # training setup 
    parser.add_argument('-mode', '--mode', type=str,choices=['from-scratch', 'KD_Hinton', 'DML', 'DIST','FitNet','FitNet-like', 'AT', 'SRD', 'SRDwithDIST', 'SRDwithKL', 'SRDwithPMSE'], help='select the type of training.')
    parser.add_argument('-dataset', '--dataset', type=str, choices=['CIFAR-10', 'CIFAR-100', 'ImageNet-1k'], help='chose dataset')
    parser.add_argument('-s_model', '--student', type=str, help='student model arch')
    parser.add_argument('-t_model', '--teacher', type=str, help='teacher model arch')
    parser.add_argument('-weight', '--weightPATH', type=str, help='path to teacher''s pretrain weights')
    # dataloader parameters 
    parser.add_argument('-train_bs','--TRAIN_BATCHSIZE',type=int, default=64, help='train batch size')
    parser.add_argument('-valid_bs','--VALID_BATCHSIZE',type=int, default=1024, help='valid batch size')
    parser.add_argument('-num_t_workers','--NUM_TRAIN_WORKER',type=int, default=16, help='number of worker threads allocated for train dataloader')
    parser.add_argument('-num_v_Workers','--NUM_VALID_WORKER',type=int, default=16, help='numner of worker threads allocated for valid dataloader')
    # solver parameters
    parser.add_argument('-epochs','--MAX_EPOCH',type=int, default=240, help='number of epochs to train for.')
    parser.add_argument('-w_decay','--W_DECAY',type=float, default=0.0005, help='SGD: weight decay value')
    parser.add_argument('-momentum','--MOMENTUM',type=float, default=0.9, help='SGD: momentum value')
    parser.add_argument('-lr','--BASE_LR',type=float, default=0.05, help='SGD: base learning rate')
    parser.add_argument('-gamma','--STEP_GAMMA',type=float, default=0.1, help='LR policy: step down factor')
    parser.add_argument('-step_s','--STEP_DOWNS' , nargs="+",type=int, default=[150,180,210], help='LR policy: Epoch(s) to step down the learning rate.')
    # distillation parameters
    parser.add_argument('-A','--ALPHA',type=float, default=0.0, help='KD weight term, alpha parameter.')
    parser.add_argument('-B','--BETA',type=float, default=0.0, help='KD weight term, beta parameter.')
    parser.add_argument('-G','--GAMMA',type=float, default=0.0, help='KD weight term, gamma parameter.')
    parser.add_argument('-T','--T',type=float, default=4, help='Ditillation temperture for vanilla KD')
    parser.add_argument('-hint','--hint_layer', type=int, default=3, help='chose hint/guided layer for FitNet')
    parser.add_argument('-pre_comp','--pre_comp', action="store_true", default=False, help='flag to enable option to precompute teacher logits (inconistent view tho.)')
    # experiment name tag (for better orgranising of results)
    parser.add_argument('-name','--Exp_Name',type=str, default='EXP_MISC', help='experiment name tag')
    
    args = parser.parse_args()
    
    # name for current run's setup 
    if args.mode == 'from-scratch':
        run_name = args.student + '_'  + args.mode + '_' + args.dataset 
    else:
        run_name = args.student + '_'  + args.mode + '_'  + args.teacher + '_' + args.dataset 
    print('\n', args.Exp_Name)    
    print('\n', run_name)

    # interface to tensorboard     
    tb_savedir = os.path.join('runs', args.Exp_Name, run_name)
    if not os.path.exists(tb_savedir):
        os.makedirs(tb_savedir)
    writer = SummaryWriter(tb_savedir)

    # dump all arguments in args (training setup + hyper-params) into tensorboard
    writer.add_text('Train Setup and Hyper-parameters',json.dumps(vars(args)))


    '''##########################################################################################################################
            Prepare chosen dataset
    ##########################################################################################################################'''
    if args.dataset == 'ImageNet-1k':
        Dataset_train= ImageNetDataset('train') # 1,281,167 train samples
        Dataset_valid = ImageNetDataset('val')  # 50k valid samples
        args.NUM_CLASS = 1000
        args.IMAGE_SIZE = 244

    elif args.dataset == 'CIFAR-100':
        Dataset_train = get_cifar100('train') # 50k train samples
        Dataset_valid = get_cifar100('test')   # 10k test samples
        args.NUM_CLASS = 100
        args.IMAGE_SIZE = 32

    elif args.dataset == 'CIFAR-10':
        Dataset_train = get_cifar10('train')  # 50k train samples
        Dataset_valid = get_cifar10('test')    # 10k test samples
        args.NUM_CLASS = 10
        args.IMAGE_SIZE = 32

    else:
        print('Chosen dataset {} not implemented. chose ''ImageNet-1k'' or ''CIFAR-100'' or ''CIFAR-10'''.format(args.dataset))
        sys.exit()

    # print number of samples in chosen datasets
    print('Chosen dataset: ', args.dataset)
    print('{} samples in train set. {} samples in test/valid set.'.format(len(Dataset_train), len(Dataset_valid)))

    # create dataloaders for train, valid and test datasets
    Dataloader_train = DataLoader(Dataset_train, batch_size = args.TRAIN_BATCHSIZE, shuffle = True, num_workers = args.NUM_TRAIN_WORKER)
    Dataloader_valid = DataLoader(Dataset_valid, batch_size = args.VALID_BATCHSIZE, num_workers = args.NUM_VALID_WORKER)
      
    # inspect a random train batch 
    #train_imgbatch, _, _ = next(iter(Dataloader_train))
    #img_grid = torchvision.utils.make_grid(train_imgbatch)
    #writer.add_image(args.dataset +': random train batch', img_grid)

    #inspect a valid batch
    #valid_imgbatch, _, _ = next(iter(Dataloader_valid))
    #img_grid = torchvision.utils.make_grid(valid_imgbatch)
    #writer.add_image(args.dataset +': random valid batch', img_grid)


    '''##########################################################################################################################
            Load the chosen model(s)
    ##########################################################################################################################'''
    model = get_Model(args.student, args.NUM_CLASS)
    if model == None:
        print('\nChosen model {} not implemented.'.format(args.student))
        sys.exit()

    # count number of model parameters student has
    total_params_s = sum(p.numel() for p in model.parameters())
    print('{} : {} model parameters'.format(args.student, total_params_s))

    # inspect to-be trained model via torchinfo 
    #summary(model, (1, 3, args.IMAGE_SIZE, args.IMAGE_SIZE,)) 
    
    # inspect to-be trained model via tensorboard
    #dummy_input = torch.randn(1, 3, args.IMAGE_SIZE, args.IMAGE_SIZE, dtype=torch.float)
    #writer.add_graph(model, dummy_input)

    teacher = None # by defualt no teacher
    
    # prepare teacher model 
    if args.teacher != None:
        teacher = get_Model(args.teacher, args.NUM_CLASS)
        
        # only load pretrain weight if not running DML where teacher is also trained
        if args.mode != 'DML': 
            # load the pre-train weights from the teacher weight PATH given 
            teacher.load_state_dict(torch.load(args.weightPATH))
            # frezee the teacher weights
            for param in teacher.parameters():
                param.requires_grad = False
                
        # count number of model parameters teacher has
        total_params_t = sum(p.numel() for p in teacher.parameters())
        print('{}(Teacher): {} model parameters'.format(args.teacher, total_params_t))
        
        # inspect teacher model via torchinfo
        #summary(teacher, (1,3,args.IMAGE_SIZE, args.IMAGE_SIZE,))
        
        # inspect teacher model via tensorboard
        #dummy_input = torch.randn(1, 3, args.IMAGE_SIZE, args.IMAGE_SIZE, dtype=torch.float)
        #writer.add_graph(teacher, dummy_input)

    '''##########################################################################################################################
            Training
    ##########################################################################################################################'''
    # setup device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # run stage-1 training if required in chosen method. (for FitNet hint based training)
    init_train(model, teacher, device, args, Dataloader_train, Dataloader_valid, writer)  
        
    # run the training!
    train(model, teacher, device, args, Dataloader_train, Dataloader_valid, writer)    
    
    '''##########################################################################################################################
            Further Evaluation
    ##########################################################################################################################'''
    # evaluate final iteration model 
    print('Evaluating the trained (final iteration) model ...')
    if args.dataset == 'CIFAR-10': # only Compute Confusion Matrix if using cifar-10 

        top1Acc, CM, Precision, Recall = evaluate(model, device, Dataloader_valid, args.NUM_CLASS, ConfMatrix = True)

        #print results
        print('Trained Model Test Acc@1: {0:.2f}'.format(top1Acc))
        print('Confusion Matrix \n',CM)
        print('Overall Recall Accuarcy: {}, Overall Precision Accuracy: {}'.format(Recall.mean(), Precision.mean()))
    
    else:
        top1Acc = evaluate(model, device, Dataloader_valid, args.NUM_CLASS, ConfMatrix = False)
        print('Trained Model Test Acc@1: {0:.2f}'.format(top1Acc))

    # also evaluate pre-trained teacher 
    if not args.teacher == None:
        print('Evaluating pre-trained teacher on test set...')
        
        if args.dataset == 'CIFAR-10':  # also compute a confusion matrix if using cifar-10 

            top1Acc, CM, Precision, Recall = evaluate(teacher, device, Dataloader_valid, args.NUM_CLASS, ConfMatrix = True)
            
            # print results 
            print('Pre-Train Teacher Test Acc@: {0:.2f}'.format(top1Acc))
            print('(Teacher) Confusion Matrix  \n',CM)
            print('(Teacher) Overall Recall Accuarcy: {}, Overall Precision Accuracy: {}'.format(Recall.mean(), Precision.mean()))
        else:
            top1Acc = evaluate(teacher, device, Dataloader_valid, args.NUM_CLASS, ConfMatrix = False)
            print('(Teacher) Test Acc@1: {0:.2f}'.format(top1Acc))


    # TODO: evlauete the KL div between distilled model's predictions aginist teacher's predictions for the images in test set.
        
    writer.flush()
    writer.close()