# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/31 15:10
# @Function: create custom logger manager
import textattack
import time
import os

from htps.AttackLogManagerV1 import AttackLogManagerV1


def parse_logger_from_args(config, train=False, epoch=0):
    # Create logger
    # attack_log_manager = textattack.loggers.AttackLogManager()
    attack_log_manager = AttackLogManagerV1()

    # Get current time for file naming
    timestamp = time.strftime("%Y-%m-%d-%H-%M")

    # Get default directory to save results
    current_dir = os.path.dirname(os.path.realpath(__file__))

    if train:
        outputs_dir = os.path.join(
            current_dir, os.pardir, 'resources', 'attack_results', 'train-attack', 'epoch'+str(epoch)
        )
    else:
        outputs_dir = os.path.join(
            current_dir, os.pardir, 'resources', 'attack_results'
        )
    out_dir_txt = out_dir_csv = os.path.normpath(outputs_dir)

    # Get default txt and csv file names
    if config.recipe:
        filename_txt = f"{config.model}_{config.recipe}_{timestamp}.txt"
        filename_csv = f"{config.model}_{config.recipe}_{timestamp}.csv"
    else:
        filename_txt = f"{config.model}_{timestamp}.txt"
        filename_csv = f"{config.model}_{timestamp}.csv"


    if config.project:
        filename_txt = f"{config.project}_{filename_txt}"
        filename_csv = f"{config.project}_{filename_csv}"

    if config.model_type:
        filename_txt = f"{config.model_type}_{filename_txt}"
        filename_csv = f"{config.model_type}_{filename_csv}"

    # if '--log-to-txt' specified with arguments
    # if config.log_to_txt:
    #     # if user decide to save to a specific directory
    #     if config.log_to_txt[-1] == "/":
    #         out_dir_txt = config.log_to_txt
    #     # else if path + filename is given
    #     elif config.log_to_txt[-4:] == ".txt":
    #         out_dir_txt = config.log_to_txt.rsplit("/", 1)[0]  # 1 represent "count"
    #         filename_txt = config.log_to_txt.rsplit("/", 1)[-1]
    #     # otherwise, customize filename
    #     else:
    #         filename_txt = f"{config.log_to_txt}.txt"

    # if "--log-to-csv" is called
    # if config.log_to_csv:
    #     # if user decide to save to a specific directory
    #     if config.log_to_csv[-1] == "/":
    #         out_dir_csv = config.log_to_csv
    #     # else if path + filename is given
    #     elif config.log_to_csv[-4:] == ".csv":
    #         out_dir_csv = config.log_to_csv.rsplit("/", 1)[0]
    #         filename_csv = config.log_to_csv.rsplit("/", 1)[-1]
    #     # otherwise, customize filename
    #     else:
    #         filename_csv = f"{config.log_to_csv}.csv"

    # in case directory doesn't exist
    # if not os.path.exists(out_dir_txt):
    #     os.makedirs(out_dir_txt)
    # if not os.path.exists(out_dir_csv):
    #     os.makedirs(out_dir_csv)

    # if "--log-to-txt" specified in terminal command (with or without arg), save to a txt file
    if config.log_to_txt == "" or config.log_to_txt:
        attack_log_manager.add_output_file(os.path.join(out_dir_txt, filename_txt))

    # if "--log-to-csv" specified in terminal command(with	or without arg), save to a csv file
    if config.log_to_csv == "" or config.log_to_csv:
        # "--csv-style used to swtich from 'fancy' to 'plain'
        color_method = None if config.csv_style == "plain" else "file"
        csv_path = os.path.join(out_dir_csv, filename_csv)
        attack_log_manager.add_output_csv(csv_path, color_method)
        textattack.shared.logger.info(f"Logging to CSV at path {csv_path}.")

    # Visdom
    if config.enable_visdom:
        attack_log_manager.enable_visdom()

    # Weights & Biases
    if config.enable_wandb:
        attack_log_manager.enable_wandb()

    # Stdout
    if not config.disable_stdout:
        attack_log_manager.enable_stdout()
    return attack_log_manager