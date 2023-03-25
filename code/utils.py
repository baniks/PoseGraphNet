import json
import shutil
import os
import torch
import logging
from collections import OrderedDict

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None, last_best=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['lr_dict'])
    
    best_val_err = None
    if last_best:
        best_val_err = checkpoint['last_best_err']
    
    return checkpoint, best_val_err


def lr_decay(optimizer, step, lr_init, decay_step, decay_rate):
    lr = lr_init * decay_rate ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_lr(optimizer):
    """get current learning rate """

    lr = 0.0
    
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    
    return lr

def copy_weight(checkpoint, model):

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']

    model_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_layers.append(name)
    
    weight_new = OrderedDict()
    copied = []
    for key, v in state_dict.items():
        if key in model_layers:
            copied.append(key)
            weight_new[key] = state_dict[key]
    
    model.load_state_dict(weight_new, strict=False)
    print("Layers copied: ", copied)


def freeze_layers(sub_modules):
    for mod in sub_modules:
        for name, param in mod.named_parameters():
            param.requires_grad = False

def write_log(log_dict_fname, log_dict_runname, log_dict, mode):

    log = {}
    if mode=="a":
        if os.path.exists(log_dict_fname):
            with open(log_dict_fname, 'r') as fp:
                old_log = json.load(fp)
                log = old_log
    log[log_dict_runname] = log_dict
    with open(log_dict_fname, 'w') as fp:
        json.dump(log, fp, sort_keys=True, indent=4)        
