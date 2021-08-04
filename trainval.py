# usage: python trainval.py -e mnist -sb results -r 1 -j None
#
# -e  [Experiment group to run like 'syn'] 
# -sb [Directory where the experiments are saved]
# -r  [Flag for whether to reset the experiments]
# -j  [Scheduler for launching the experiment like 'slurm, toolkit, gcp'. 
#      None for running them on local machine]


import tqdm
import argparse
import os

from haven import haven_wizard as hw
from haven import haven_results as hr

from src import datasets, models
import tensorflow as tf
import numpy as np


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Create data loaders and model
    # ==============================
    train_loader = datasets.get_loader(name=exp_dict["dataset"], 
                                       split="train", 
                                       datadir=os.path.dirname(savedir), 
                                       exp_dict=exp_dict)

    valid_loader = datasets.get_loader(name=exp_dict["dataset"], 
                                       split="val", 
                                       datadir=os.path.dirname(savedir), 
                                       exp_dict=exp_dict)
    
    test_loader = datasets.get_loader(name=exp_dict["dataset"], 
                                      split="test", 
                                      datadir=os.path.dirname(savedir), 
                                      exp_dict=exp_dict)

    model = models.get_model(name=exp_dict["model"], exp_dict=exp_dict)

    # Resume or initialize checkpoint
    # ===============================
    cm = hw.CheckpointManager(savedir)
    # todo: add support for resuming from a checkpoint
    # 
    # pytorch:
    # state_dict = cm.load_model()
    # if state_dict is not None:
    #     model.set_state_dict(state_dict)
    # 
    # for keras can we do something like:
    # model = cm.load_model(model_type="keras")

    # Train and Validate
    # ===================
    # todo: check if we can resume from `cm.get_epoch()` as in pytorch.
    for epoch in tqdm.tqdm(range(0, exp_dict['epochs']), desc="Running Experiment"):
        epoch_losses_train, epoch_accs_train = [], []
        for batch_idx, (xs_batch, ys_batch) in enumerate(train_loader):
            dic = model.train_on_batch(xs_batch, ys_batch, sample_weight=None, class_weight=None, reset_metrics=True, return_dict=True,)

            epoch_losses_train.append(dic['loss'])
            epoch_accs_train.append(dic['accuracy'])
    
            # need to break as the train loader is infinite
            if batch_idx > len(train_loader):
                break

        val_stats = model.evaluate_generator(valid_loader, steps=len(valid_loader))
        acc_train, loss_train = np.mean(epoch_accs_train), np.mean(epoch_losses_train) 
        acc_valid, loss_valid = val_stats[1], val_stats[0]

        cm.log_metrics(dict(epoch=epoch, acc_train=acc_train, loss_train=loss_train, acc_valid=acc_valid, loss_valid=loss_valid))
        
        # todo: cm.save_keras? like `cm.save_torch("model.pth", model.state_dict())`
        # cm.save_pkl("model.pkl", model) causes TypeError: can't pickle weakref objects

    print("Experiment done!!\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb", "--savedir_base", required=True, help="Define the base directory where the experiments will be saved."
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("-j", "--job_scheduler", default=None, help="Choose Job Scheduler.")
    parser.add_argument("--python_binary", default='python', help='path to your python executable')

    args, others = parser.parse_known_args()

    # Define a list of experiments
    if args.exp_group == "mnist":
        exp_list = []
        for lr in [1, 1e-2, 1e-4,]:
            exp_list += [{"dataset": "mnist", "model": "convnet_shallow",
                          "seed": 111,
                          "epochs": 5, "batch_size": 128,
                          "optimizer": "adam", "optimizer_params": {"lr": lr},
                          "augmentations": {"rescale": 1/255.,},
                          "metrics": ["accuracy"],
                          "loss": "categorical_crossentropy",}]

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "slurm":
        job_config = {
            "account_id": "def-dnowrouz-ab",
            "time": "1:00:00",
            "cpus-per-task": "2",
            "mem-per-cpu": "20G",
            "gres": "gpu:1",
        }

    elif args.job_scheduler == "toolkit":
        import job_configs
        job_config = job_configs.JOB_CONFIG

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results.ipynb",
        python_binary_path=args.python_binary,
        args=args
    )