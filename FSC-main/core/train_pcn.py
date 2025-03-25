import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_pcn import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import *
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from models.model_utils import PCViews
# from models.SVDFormer import Model
from models.FSCSVD import Model
# from models.FFSC import Model
import wandb # log print

def setup_logging(log_file_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    wandb.init(project="FSC", config=cfg)
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after the first run

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
    if not os.path.exists(cfg.DIR.LOGS):
        os.makedirs(cfg.DIR.LOGS)

    log_file_path = os.path.join(cfg.DIR.LOGS, 'training.log')
    setup_logging(log_file_path)

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)
    


    # # Set up folders for logs and checkpoints
    # output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    # cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    # cfg.DIR.LOGS = output_dir % 'logs'
    # if not os.path.exists(cfg.DIR.CHECKPOINTS):
    #     os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # model = Model(cfg)
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()

    # FFSC
    # # model load
    # pretrained_path = 'ckpt-best.pth'
    # pretrained_dict = torch.load(pretrained_path)

    model = Model(cfg)
    # model pretrain
    # model.load_state_dict(pretrained_dict, strict=False)
    # #  refine2 
    # refine2_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('refine2.')}

    # # load refine2 
    # model.refine2.load_state_dict(refine2_dict, strict=False)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # Watch the model
    if not wandb.watch_called:
        wandb.watch(model, log="all")
        wandb.watch_called = True

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0
    # # SVDformer
    # render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                # # SVDformer
                # partial_depth = torch.unsqueeze(render.get_img(partial), 1)
                # pcds_pred = model(partial, partial_depth)
                
                # FSC
                pcds_pred = model(partial)

                loss_total, losses = get_loss(pcds_pred,  gt, sqrt=True)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)

                # Log to wandb
                wandb.log({"cd_pc": cd_pc_item, "cd_p1": cd_p1_item, "cd_p2": cd_p2_item})

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)

        # Log epoch averages to wandb
        wandb.log({"avg_cd_pc": avg_cdc, "avg_cd_p1": avg_cd1, "avg_cd_p2": avg_cd2})

        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2]]))

        # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    train_writer.close()
    val_writer.close()
    wandb.finish() 
