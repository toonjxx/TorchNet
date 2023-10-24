from utils import str2bool
import torch
from datasets import create_data_loader,train_val_dataloader_kfold
import models.convnext as convnext
from trainers import Trainer_Hyper, Trainer_BP
import torch.optim.lr_scheduler
import time
import argparse
from torch.backends import cudnn
import wandb


def get_args_parser():

    # Load settings from the config file
    parser = argparse.ArgumentParser('ConvNeXT training', add_help=False)

    # Add parser for device
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')


    # Add parser for model config
    parser.add_argument
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels')
    parser.add_argument('--num_class', type=int, default=1, help='number of classes')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', default='Convnext_ShallowPico2', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.)')
    parser.add_argument('--modeltype', type=str, default='classification', help='classification or regression')


    # Add parser for model training config
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--k_fold', type=int, default=5, help='k for kfold')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='early stop patience')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='lr scheduler')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.75, help='lr scheduler gamma')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=10, help='lr scheduler step size')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')



    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--seed', default=0, type=int)

    # Add parser for data
    parser.add_argument('--data_dir', type=str, default='../Data', help='data directory')

    return parser

def crossvalidation(args):
    
    dataloaders = train_val_dataloader_kfold("Train", batch_size=args.batch_size, k=args.k_fold)
    models = []
    trainers = []
    timenow = time.time()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True


    for fold_idx, (train_dataloader, val_dataloader) in enumerate(dataloaders):
        model = convnext.__dict__[args.model](
        drop_path_rate=args.drop_path
        )
        model = model.to(device)
        models.append(model)

        checkpoint_dir = f"Checkpoints/val_{timenow}/fold{fold_idx+1}"
        log_dir = f"logs/val_{timenow}/fold{fold_idx+1}"

        trainer = Trainer_Hyper(model, train_dataloader, val_dataloader, num_epochs=200 ,log_dir=log_dir,checkpoint_dir=checkpoint_dir,early_stop_patience=20)
        trainers.append(trainer)
    val_acc = []
    train_acc = []
    for i in range(args.k_fold):
        print(f"Training model {i+1}/{args.k_fold}")
        train,val=trainers[i].train()
        val_acc.append(val)
        train_acc.append(train)
    print(f"Average train accuracy: {sum(train_acc)/args.k_fold}")
    print(f"Average val accuracy: {sum(val_acc)/args.k_fold}")

def train(args):
    timenow = time.time()
    device = torch.device(args.device)

    test_dataloader = create_data_loader("Test", batch_size=args.batch_size)
    train_dataloader = create_data_loader("Train", batch_size=args.batch_size)
    model_name = 'ConvNeXt1d_L_Hypertension'
    checkpoint_dir = f"Checkpoints/train_{model_name}_{timenow}"
    log_dir = f"logs/train_{model_name}_{timenow}"

    model = convnext.__dict__[args.model](
        drop_path_rate=args.drop_path
        )
    model = model.to(device)
    trainer = Trainer_Hyper(model, train_dataloader, test_dataloader, num_epochs=500,log_dir=log_dir,checkpoint_dir=checkpoint_dir,early_stop_patience=50)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ConvNeXt training', parents=[get_args_parser()])
    wandb.login()

    args = parser.parse_args()
    crossvalidation(args)
    #train(args)