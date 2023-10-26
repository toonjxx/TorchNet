from utils import str2bool,load_config
import torch
from datasets import create_data_loader,train_val_dataloader_kfold
from engine import Engine
import torch.optim.lr_scheduler
import time
from torch.backends import cudnn
import wandb
from models.convnext import ConvNeXt


def train():

    wandb.init(project="PPG_convnext")
    args = wandb.config
    dataloaders = train_val_dataloader_kfold("Train", args)
    engines = []

    timenow = time.time()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    val_acc = []
    train_acc = []

    for fold_idx, (train_dataloader, val_dataloader) in enumerate(dataloaders):
        model = ConvNeXt(depths=args["depths"], dims=args["dims"],in_chans=args["num_channels"],
                         num_classes=args["num_class"],drop_path_rate=args["drop_path"],
                         dim_mul=2,dwconv_kernel_size=3,dwconv_padding=1,
                         downsample_stem=2).to(device)
        model = model.to(device)
        if wandb.run is None:
            id = timenow
        else:
            id = wandb.run.id
        checkpoint_dir = f"{args.checkpoint_dir}/{id}/val_{timenow}/fold{fold_idx+1}"
        log_dir = f"logs/{id}/val_{timenow}/fold{fold_idx+1}"
        engine = Engine(model, train_dataloader, val_dataloader, args=args, log_dir = log_dir,checkpoint_dir=checkpoint_dir)
        engines.append(engine)

    for i in range(args.repeat):
        print(f"Training model {i+1}/{args.repeat}")
        train,val=engines[i].train()
        val_acc.append(val)
        train_acc.append(train)

    wandb.log({"train_acc":train_acc,"val_acc":val_acc,"avg_train_acc":sum(train_acc)/args.repeat,"avg_val_acc":sum(val_acc)/args.repeat})

    print(f"Average train accuracy: {sum(train_acc)/args.repeat}")
    print(f"Average val accuracy: {sum(val_acc)/args.repeat}")
    print(f"train accuracy: {train_acc}")
    print(f"val accuracy: {val_acc}")

if __name__ == "__main__":

    wandb.login()
    sweep_config = load_config("Hypertension_sweep_config.yaml")
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="PPG_convnext")
    wandb.agent(sweep_id=sweep_id, function=train)

 