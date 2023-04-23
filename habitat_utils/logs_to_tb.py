import glob
import os
import os.path as osp
import time

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(logs_dir, replace, tb_name):
    tb_dir = logs_to_tb_dir(logs_dir, tb_name)
    if osp.exists(tb_dir):
        # Confirm with user before deleting existing directory
        prompt = f"Data in directory {tb_dir} already exists. Delete? [y/n]: "
        if replace or input(prompt) == "y":
            print(f"Deleting data in {tb_dir}")
            for event in glob.glob(osp.join(tb_dir, "events.out.tfevents.*")):
                os.remove(event)
        else:
            print("Exiting.")
            return
    print(f"Writing to {osp.abspath(tb_dir)}")
    writer = SummaryWriter(log_dir=tb_dir)
    log_files = glob.glob(osp.join(logs_dir, "*.log"))
    log_files.sort(key=lambda x: int(x.split(".")[-2]))
    print(f"Found {len(log_files)} log files.")
    count = 0
    for log_file in tqdm.tqdm(log_files):
        step_id, aggregated_stats = log_to_stats(log_file)
        if len(aggregated_stats) == 0:
            print(f"Skipping {log_file} because it has no stats.")
            continue
        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, step_id)
        count += 1
    time.sleep(3)  # Need to wait for the writer to finish writing... =_=
    print(f"Successfully plotted {count} log files.")


def log_to_stats(log_file):
    with open(log_file, "r") as f:
        log_contents = f.read()

    # Check if the step id is present in the file. If not, we have to load the
    # corresponding checkpoint
    if "step_id: " in log_contents:
        step_id = int(log_contents.split("step_id: ")[1].split("\n")[0])
    else:
        grandparent_dir = osp.dirname(osp.dirname(log_file))
        ckpt_basename = osp.basename(log_file).replace(".log", ".pth")
        candidates = glob.glob(osp.join(grandparent_dir, f"*/{ckpt_basename}"))
        assert (
            len(candidates) == 1
        ), f"Found {len(candidates)} candidates for {ckpt_basename}"
        ckpt_file = candidates[0]
        ckpt = torch.load(ckpt_file, map_location="cpu")
        step_id = ckpt["extra_state"]["step"]
        # Add it to the file so we don't have to load the checkpoint again
        with open(log_file, "a") as f:
            f.write(f"\nstep_id: {step_id}\n")

    lines = log_contents.splitlines()
    aggregated_stats = {}
    for line in lines:
        if "Average episode " in line:
            key = line.split("Average episode ")[1].split(":")[0]
            value = float(line.split(": ")[-1])
            aggregated_stats[key] = value
    return step_id, aggregated_stats


def logs_to_tb_dir(log_dir, base_dir="tb_eval"):
    parent_dir = osp.dirname(osp.abspath(log_dir))
    tb_dir = osp.join(parent_dir, base_dir)
    return tb_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", type=str)
    parser.add_argument("-y", "--replace", action="store_true")
    parser.add_argument(
        "-t",
        "--tb-name",
        help="Name of the tensorboard log directory (default=tb_eval)",
        default="tb_eval",
    )
    args = parser.parse_args()
    main(args.logs_dir, args.replace, args.tb_name)
