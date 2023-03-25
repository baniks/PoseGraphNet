import numpy as np
import argparse
import time
from time import strftime, gmtime
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json

from h36m_dataset import Human36M
import utils as utils
from evaluate import mpjpe, p_mpjpe
from model_gct import GCNResidualBN_adjL, weight_init, count_parameters
import loss 
from graph import Graph

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="../../datasets/human3.6m/orig/pose",
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../models',
                    help="Directory containing params.json")
parser.add_argument('--init_weight', default=None,
                    help="Path of the init weight file")
parser.add_argument('--restore_file', default=None, 
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--tf_logdir', default='../runs/',
                    help="Optional, Directory for saving tensorboard logs.")

parser.add_argument('--exp', default="icip", help="experiment name")
parser.add_argument('--exp_suffix', default="", help="experiment number")
parser.add_argument('--run_suffix', default="", help="run number")
parser.add_argument('--exp_desc', default="", help="experiment description")
parser.add_argument('--cpn_file', default="", help="cpn file")
parser.add_argument('--test', action=argparse.BooleanOptionalAction, help='train or test mode')
parser.add_argument('--checkpoint', default="", help="checkpoint file")


def write_train_summary_scalars(writer, experiment, epoch, batch_idx, num_batches, summary):
    writer.add_scalar("loss/train_loss", summary['train_loss'], epoch)
    writer.add_scalar("accuracy/train_mpjpe_err", summary['train_mpjpe_err'], epoch)

def write_val_summary_scalars(writer, experiment, epoch, batch_idx, num_batches, summary_epoch, detailed=False):
    writer.add_scalar("loss/val_loss", summary_epoch['val_loss'], epoch)
    writer.add_scalar("accuracy/val_mpjpe_err", summary_epoch['val_mpjpe_err'], epoch)
    writer.add_scalar("accuracy/val_err_x", summary_epoch['val_err_x'], epoch)
    writer.add_scalar("accuracy/val_err_y", summary_epoch['val_err_y'], epoch)
    writer.add_scalar("accuracy/val_err_z", summary_epoch['val_err_z'], epoch)        

    if detailed:
        writer.add_scalars("accuracy/val_joint_err", {
            #'Hip': summary_epoch['Hip'],
            'RHip': summary_epoch['RHip'],
            'LHip': summary_epoch['LHip'],
            'RKnee': summary_epoch['RKnee'],
            'LKnee': summary_epoch['LKnee'],
            'RFoot': summary_epoch['RFoot'],
            'LFoot': summary_epoch['LFoot'],
            'Spine': summary_epoch['Spine'],
            'Thorax': summary_epoch['Thorax'],
            'Neck': summary_epoch['Neck'],
            'Head': summary_epoch['Head'],
            'LShoulder': summary_epoch['LShoulder'],
            'RShoulder': summary_epoch['RShoulder'],
            'LElbow': summary_epoch['LElbow'],
            'RElbow': summary_epoch['RElbow'],
            'LWrist': summary_epoch['LWrist'],
            'RWrist': summary_epoch['RWrist']
        } , epoch)

# def write_adjacency(writer, adj, epoch):
#     fig = plot_adjacency_matrix(adj)
#     writer.add_figure("adj", fig, epoch)


def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, writer, log_dict, exp, adj):
    t = time.time()
    summary = []
    
    model.train()
    optimizer.zero_grad()
    
    num_batches = len(dataloader)

    for i_batch, data in enumerate(dataloader):

        # load and process data from DL
        batch_inputs = data['inputs']
        batch_gt = data['outputs']

        if params.cuda:
            batch_inputs = batch_inputs.cuda()
            batch_gt = batch_gt.cuda()
        
        batch_inputs = Variable(batch_inputs)
        batch_gt = Variable(batch_gt)
        
        # output = model(batch_inputs, adj)
        output = model(batch_inputs)

        loss_train = loss_fn(output, batch_gt)
                
        # update model
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if params.with_hip:
            output[:, 0, :] = 0
            
        err_train, _ = metrics[0](output, batch_gt)
        gc1_grad = 0
                
        summary_batch = {
            'train_loss': loss_train.item(),
            'train_mpjpe_err': err_train.item(), 
            'gc1_grad': gc1_grad
            }

        summary.append(summary_batch)
        
        if i_batch % params.save_summary_steps == 0:
            
            print('Epoch: {:04d}'.format(epoch),
                  'Batch: {}/{}'.format(i_batch+1, num_batches),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'train_mpjpe_err: {:.4f}'.format(err_train.item()),
                  # 'lr_now: {:.6f}'.format(lr_now)
                  'time: {:.4f}s'.format(time.time() - t)
                  ) 
    
    # mean metrics
    metrics_loss_mean = np.mean([x['train_loss'] for x in summary])
    metrics_acc_mean = np.mean([x['train_mpjpe_err'] for x in summary], axis=0)
    metrics_loss_mean_str = "loss: {0:5.7f} ".format(np.mean([x['train_loss'] for x in summary]))
    metrics_acc_mean_str = "avg_mpjpe_err: {0:5.3f} ".format(np.mean([x['train_mpjpe_err'] for x in summary], axis=0))
    
    # Logging
    # for log file
    logging.info(
        "- Train metrics -\t"+
          "Epoch: " + str(epoch) + "\t" +
          metrics_loss_mean_str + "\t" +
          metrics_acc_mean_str
          )
    
    # for tensorboard
    summary_epoch = {
        'train_loss': metrics_loss_mean,
        'train_mpjpe_err': metrics_acc_mean
    }
    write_train_summary_scalars(writer=writer,
        experiment=exp,
        epoch=epoch,
        batch_idx=i_batch,
        num_batches=num_batches,
        summary=summary_epoch)

    # for log dict
    log_dict['train_losses'].append(metrics_loss_mean)
    log_dict['train_errors'].append(metrics_acc_mean)

    return metrics_loss_mean

    
def evaluate(model, loss_fn, dataloader, metrics, params, epoch, writer, log_dict, exp, adj, detailed=True, viz=True, joint_dict=None):
    t = time.time()
    model.eval()
    summary = []
    _lastOutput = None
    _lastGt = None
    _lastBatch = None

    num_batches = len(dataloader)
    err_val_p2_list = []

    for i_batch, data in enumerate(dataloader):
        
        # load and process data from DL
        batch_inputs = data['inputs']
        batch_gt = data['outputs']

        if params.cuda:
            batch_inputs = batch_inputs.cuda()
            batch_gt = batch_gt.cuda()
        
        batch_inputs = Variable(batch_inputs)
        batch_gt = Variable(batch_gt)
        
        # output = model(batch_inputs, adj)
        output = model(batch_inputs)
    
        if i_batch == (num_batches-1):
            _lastOutput = output
            _lastGt = batch_gt
            _lastBatch = i_batch
                
        loss_val = loss_fn(output, batch_gt)

        if params.with_hip:
            output[:, 0, :] = 0

        err_val, err_joint_val = metrics[0](output, batch_gt)
        if len(metrics)>1:
            err_val_p2 = metrics[1](output.cpu().data.numpy(), batch_gt.cpu().data.numpy())
            err_val_p2_list.append(err_val_p2)

        err_val_x, err_joint_val_x = metrics[0](output[:, :, :1] , batch_gt[:,:,:1])
        err_val_y, err_joint_val_y = metrics[0](output[:, :, 1:2] , batch_gt[:,:,1:2])
        err_val_z, err_joint_val_z = metrics[0](output[:, :, 2:3] , batch_gt[:,:,2:3])
                        
        summary_batch = {
            'val_loss': loss_val.item(),
            'val_mpjpe_err': err_val.item(),
            'val_err_x': err_val_x.item(),
            'val_err_y': err_val_y.item(),
            'val_err_z': err_val_z.item(),
            'val_mpjpe_err_joint': err_joint_val.cpu().data.numpy(),
            'val_err_x_joint': err_joint_val_x.cpu().data.numpy(),
            'val_err_y_joint': err_joint_val_y.cpu().data.numpy(),
            'val_err_z_joint': err_joint_val_z.cpu().data.numpy()
            }
        summary.append(summary_batch)

    # mean metrics
    metrics_loss_mean = np.mean([x['val_loss'] for x in summary])
    metrics_acc_mean = np.mean([x['val_mpjpe_err'] for x in summary], axis=0)
    metrics_acc_x_mean = np.mean([x['val_err_x'] for x in summary], axis=0)
    metrics_acc_y_mean = np.mean([x['val_err_y'] for x in summary], axis=0)
    metrics_acc_z_mean = np.mean([x['val_err_z'] for x in summary], axis=0)
    
    metrics_loss_mean_str = "loss: {0:5.7f} ".format(metrics_loss_mean)
    metrics_acc_mean_str = "avg_mpjpe_err: {0:5.3f} ".format(metrics_acc_mean)
    metrics_acc_x_mean_str = "avg_err_x: {0:5.3f} ".format(metrics_acc_x_mean)
    metrics_acc_y_mean_str = "avg_err_y: {0:5.3f} ".format(metrics_acc_y_mean)
    metrics_acc_z_mean_str = "avg_err_z: {0:5.3f} ".format(metrics_acc_z_mean)

    # Log entries
    # for log file
    logging.info("- Val metrics -\t"+
          "Epoch: " + str(epoch) + "\t" +
          metrics_loss_mean_str + "\t" +
          metrics_acc_mean_str + "\t" +
          metrics_acc_x_mean_str + "\t" +
          metrics_acc_y_mean_str + "\t" +
          metrics_acc_z_mean_str
        )
    
    if len(metrics)>1:
        err_val_p2_list = np.asarray(err_val_p2_list)
        print(err_val_p2_list.shape)
        print("- Val metrics -\t"+
            "Epoch: " + str(epoch) + "\t" +
            "avg_mpjpe_P2_err: {0:5.3f}".format(np.mean(err_val_p2_list))
        )
    
    # for tensorboard
    summary_epoch = {
        'val_loss': metrics_loss_mean,
        'val_mpjpe_err': metrics_acc_mean,
        'val_err_x': metrics_acc_x_mean,
        'val_err_y': metrics_acc_y_mean,
        'val_err_z': metrics_acc_z_mean
    }
    
    # for log dict
    if log_dict:
        log_dict['val_losses'].append(metrics_loss_mean)
        log_dict['val_errors'].append(metrics_acc_mean)
        log_dict['val_errors_x'].append(metrics_acc_x_mean)
        log_dict['val_errors_y'].append(metrics_acc_y_mean)
        log_dict['val_errors_z'].append(metrics_acc_z_mean)   

    # joint level scores
    metrics_err_joint = np.mean([x['val_mpjpe_err_joint'] for x in summary], axis=0).astype(np.float64)  
    metrics_err_joint_x = np.mean([x['val_err_x_joint'] for x in summary], axis=0)   
    metrics_err_joint_y = np.mean([x['val_err_y_joint'] for x in summary], axis=0)   
    metrics_err_joint_z = np.mean([x['val_err_z_joint'] for x in summary], axis=0)   

    metrics_joint = "Joint:,Mean,Mean_X,Mean_Y,Mean_Z\n"
    for joint_id in range(len(metrics_err_joint)):
        # for log file
        metrics_joint += "{0},{1:5.2f},{2:5.2f},{3:5.2f},{4:5.2f}\n".format(
            joint_dict[joint_id],
            metrics_err_joint[joint_id],
            metrics_err_joint_x[joint_id],
            metrics_err_joint_y[joint_id],
            metrics_err_joint_z[joint_id]
            )

        # for tensorboard
        summary_epoch[joint_dict[joint_id]]=metrics_err_joint[joint_id]
        # for log dict
        if log_dict:
            log_dict['val_errors_joints'][joint_dict[joint_id]].append(metrics_err_joint[joint_id])
    
    logging.info("- Val metrics Joint-\t"+
        "Epoch: " + str(epoch) + "\n" +
        metrics_joint
        )

    # Update tensorboard writer
    if writer:
        write_val_summary_scalars(writer=writer,
                            experiment=exp,
                            epoch=epoch,
                            batch_idx=i_batch,
                            num_batches=num_batches,
                            summary_epoch=summary_epoch,
                            detailed=detailed)    
            
    return {'val_err': np.mean([x['val_mpjpe_err'] for x in summary], axis=0)}
    

def train_and_evaluate(model, train_dataloader, val_dataloader, adj, optimizer, scheduler, loss_fn, metrics, params,
                       model_dir, tf_logdir, log_dict, exp, viz, restore_file=None, joint_dict=None):
    """
    Train the model and evaluate every epoch
    :return:
    """
    best_val_err = params.last_best_err
    log_dict['best_val_err'] = { "epoch": "-", "err": best_val_err }

    # reload weights from restore_file if specified
    if restore_file:
        logging.info("Restoring parameters from {}".format(restore_file))
        _, best_val_err = utils.load_checkpoint(restore_file, model, optimizer, scheduler, last_best=True)
        log_dict['best_val_err'] = { "epoch": "-", "err": best_val_err }
        logging.info("- done.")

    logging.info("last_best_err: "+ str(log_dict['best_val_err']['err']))

    # forced change of lr
    if params.force_lr_change:
        for param_group in optimizer.param_groups:
            param_group['lr']=params.learning_rate
            logging.info("Forced lr change - start lr: " + str(param_group['lr']))

    writer = SummaryWriter(tf_logdir)
    end_epoch = params.start_epoch + params.num_epochs

    log_dict['train_losses'] =[]
    log_dict['train_errors'] =[]
    log_dict['val_losses'] =[]
    log_dict['val_errors'] = []
    log_dict['val_errors_x'] = []
    log_dict['val_errors_y'] = []
    log_dict['val_errors_z'] = []
    log_dict['val_errors_joints'] = {}

    for joint_id in range(params.num_joints):
        log_dict['val_errors_joints'][joint_dict[joint_id]] = []

    for epoch in range(params.start_epoch, end_epoch):
        logging.info("Epoch {}/{}".format(epoch, end_epoch-1))
        t0 = time.time()
        # train for one epoch
        train_loss = train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, writer, log_dict, exp, adj)
        logging.info("Epoch {} training time: {}".format(epoch, strftime("%H:%M:%S", gmtime(time.time() - t0))))
        lr_now = utils.get_lr(optimizer)
        logging.info("Epoch {} learning_rate: {:.10f}".format(epoch, lr_now))
        log_dict['hyperparameters']['lr'].append(round(lr_now, 10))

        is_best = False

        # evaluate every epoch, on the validation set
        if epoch%1 == 0:
            t1 = time.time()
            val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, epoch, writer, log_dict, exp, adj, detailed=True, viz=viz, joint_dict=joint_dict)
            logging.info("Epoch {} validation time: {}".format(epoch, strftime("%H:%M:%S", gmtime(time.time() - t1)))) 
            val_err = val_metrics['val_err']

            # update lr
            scheduler.step(train_loss)

            if epoch>params.start_epoch:
                is_best = val_err <= best_val_err
            elif epoch == 0:
                best_val_err = val_err
            
            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new min validation error")
                best_val_err = val_err
                log_dict['best_val_err'] = { "epoch": epoch, "err": best_val_err }
        
        # # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'lr_dict': scheduler.state_dict(),
                               'last_best_err': best_val_err 
                               },
                              is_best=is_best,
                              checkpoint=model_dir)
        
        # save temporary run log 
        utils.write_log(log_dict_run_fname, log_dict_runname, log_dict, "w")

    writer.close()


def main():

    ##################################################################
    # Experiment setup
    ##################################################################
   
    # Parse arguments
    args = parser.parse_args()

    exp = args.exp
    exp_suffix = args.exp_suffix
    run_suffix = args.run_suffix
    exp_desc = args.exp_desc

    tb_logdir = args.tf_logdir + exp + '/' + exp_suffix + "_" + run_suffix
    model_dir = args.model_dir + "/" + exp + '/' + exp_suffix
    train_test = "test" if args.test else "train"

    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)

    # Load parameters
    json_path = os.path.join('../models/', exp, exp_suffix, 'params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # CUDA settings
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed_value = 230
    if params.cuda:
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(seed_value)
    else:
        device = torch.device("cpu")
        torch.manual_seed(seed_value)

    ##################################################################
    # Logger creation
    ##################################################################
    
    utils.set_logger(os.path.join(model_dir, exp + '_' + exp_suffix + ".log"))
    logging.info("##################################################################")
    logging.info("Experiment: " + exp + '_' + exp_suffix + '_' + run_suffix)
    logging.info("Train/test: " + train_test)
    logging.info("Description: "+ exp_desc)
    logging.info(
        "Parameters:\tlearning rate: " + str(params.learning_rate) +
        "\tbatch_size: " + str(params.batch_size) +
        "\tepochs: " + str(params.start_epoch) + " - " + str(params.start_epoch + params.num_epochs) +
        "\tdrop out: " + str(params.dropout_rate)
        )

    global log_dict_fname, log_dict_runname, log_dict_run_fname, log_dict
    log_dict_fname = os.path.join(model_dir, f"{exp + '_' + exp_suffix}.json")
    log_dict_runname = exp + '_' + exp_suffix + '_' + run_suffix
    log_dict_run_fname = os.path.join(model_dir, f"{log_dict_runname}.json")

    log_dict = {}
    log_dict['exp_name'] = exp + '_' + exp_suffix + '_' + run_suffix
    log_dict['exp_desc'] = exp_desc
    log_dict['tensorboard_logdir'] = tb_logdir
    log_dict['hyperparameters']= {
        'lr': [],
        'batch_size': params.batch_size,
        'start_epoch' : params.start_epoch,
        'end_epoch' : params.start_epoch + params.num_epochs,
        'dropout' : params.dropout_rate,
        "seed": seed_value,
        "loss" : "mpjpe"
    }
    log_dict['num_blocks']=params.num_blocks  


    ##################################################################
    # Dataset loading
    ##################################################################
    
    if train_test=="train":
        logging.info("Loading training dataset....")
        train_dataset = Human36M(data_dir=args.data_dir, cpn_file=args.cpn_file, train=True, with_hip=params.with_hip, ds_category=params.ds_category)
        logging.info("Train dataset len: {}".format(len(train_dataset)))
        train_dl = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True)
        logging.info("Number of training batches: {}".format(len(train_dl)))
        logging.info("- done.")

    logging.info("Loading test dataset....")
    test_dataset = Human36M(data_dir=args.data_dir, cpn_file=args.cpn_file, train=False, with_hip=params.with_hip, ds_category=params.ds_category) 
    logging.info("Test dataset len : {}".format(len(test_dataset)))
    val_dl = DataLoader(
            test_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True)    
    logging.info("Number of validation batches: {}".format(len(val_dl)))
    logging.info("- done.")

    # joint id to name mapping
    if params.with_hip:
        joint_id_to_names = test_dataset.get_h36m_joint_id_to_name()
        joint_name_to_id = test_dataset.get_h36m_joint_name_to_id()
    else:
        joint_id_to_names = test_dataset.get_h36m_joint_id_to_name_ex_hip()
        joint_name_to_id = test_dataset.get_h36m_joint_name_to_id_ex_hip()

    ##################################################################
    # Define model 
    ##################################################################
    logging.info("Loading model:")
    logging.info("num of output classes: "+ str(params.num_joints))

    a = Graph(layout='hm36_gt', strategy='spatial', with_hip=params.with_hip, norm=False)
    adj = torch.from_numpy(a.A).float()  # split adjacency

    adjsize_list = list(adj.size())
    adjsize_str = ",".join([str(x) for x in adjsize_list])
    logging.info("Adjacency matrix shape: "+ adjsize_str)

    if params.cuda:
        adj = adj.cuda(device)
        
    logging.info("Adjacency matrix loaded.")

    model = GCNResidualBN_adjL(nfeat=params.input_feat,
                   nhid=params.num_hidden,
                   nclass=params.output_feat,
                   dropout=params.dropout_rate,
                   num_joints=params.num_joints,
                   adj=adj,
                   num_groups=3
                   )

    logging.info("Num of parameters: " +str(count_parameters(model)))

    for name, p in model.named_parameters():
        if p.requires_grad:
            psize_list = list(p.size())
            psize_str = [str(x) for x in psize_list]
            psize_str = ",".join(psize_str)
            logging.info(name+ "\t"+ psize_str)
                
    if params.cuda:
        model = model.cuda(device)
    
    logging.info("- done.")

    if params.init_weights:
        model.apply(weight_init)
        utils.copy_weight(params.init_weights, model)
        log_dict['init_weights']=params.init_weights
        logging.info("Model initialized from " + params.init_weights)
    else:
        model.apply(weight_init)
        logging.info("Model initialized")

    ##################################################################
    # Define optimizer
    ##################################################################

    logging.info("Creating Optimizer....")
    optimizer = optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        threshold=1e-4
    )
    logging.info("- done.")
    logging.info("Learning rate: {}".format(params.learning_rate))
    #logging.info("Weight decay: {}".format(params.weight_decay))

    ##################################################################
    # fetch loss function and metrics
    ##################################################################
    loss_fn = loss.mpjpe_loss()

    if params.cuda:
        loss_fn = loss_fn.cuda(device)

    metrics = [mpjpe, p_mpjpe]

    ##################################################################
    # Training
    ##################################################################
    if train_test == "train":
        logging.info("Starting training.. ")
        train_and_evaluate(model, train_dl, val_dl, adj, optimizer, scheduler, loss_fn, metrics, params, model_dir, tb_logdir, log_dict,
                        exp, False, params.restore_file, joint_id_to_names)

        # save log file
        utils.write_log(log_dict_fname, log_dict_runname, log_dict, "a")
        logging.info("Training finished.")
        logging.info("Log saved to " + log_dict_fname)
        try:
            os.remove(log_dict_run_fname)
            logging.info("run log file deleted: " + log_dict_run_fname)
        except OSError as error:
            logging.info("run log file not deleted "+ log_dict_run_fname)

    ##################################################################
    # Validation
    ##################################################################
    if train_test == "test":
        logging.info("Evaluating {}".format(exp))

        checkpoint = args.checkpoint
        assert os.path.isfile(checkpoint), "No checkpoint file found at {}".format(checkpoint)
            
        logging.info("Restoring from {}".format(checkpoint))
        utils.load_checkpoint(checkpoint, model, optimizer)
        logging.info("- done.")
        val_metrics = evaluate(model, loss_fn, val_dl, metrics, params, epoch=0, writer=None, log_dict=None, exp=exp, adj=adj, detailed=True, viz=False, joint_dict=joint_id_to_names)

if __name__ == "__main__":
    main()

