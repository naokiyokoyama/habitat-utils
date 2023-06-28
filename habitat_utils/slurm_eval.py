"""
Given a checkpoint directory and a template slurm script, this script will submit jobs
to evaluate all checkpoints until all checkpoints have been evaluated. The user can
provide a minimum number of checkpoints as well, which will make this script wait for
more checkpoints before terminating. In such a case, this script is best run inside
a tmux session.

The given template slurm script will ONLY be altered in the following ways:
1. The string "${SLURM_CHECKPOINT_PATH}" will be replaced with the path to a checkpoint
1. The string "${SLURM_LOG_PATH}" will be replaced with the path to the log file, which
will be inferred from the given checkpoint directory

It's the user's responsibility to make sure that the rest of the template slurm script
produces the desired behavior.
"""


import argparse
import glob
import os
import os.path as osp
import subprocess
import time
from datetime import datetime

from habitat_utils.logs_to_tb import main as logs_to_tb

THIS_DIR = osp.dirname(osp.abspath(__file__))
SINGLE_CKPT_TEMPLATE = osp.join(THIS_DIR, "single_ckpt_eval.sh")

TMP_SLURM_FILE = osp.join(THIS_DIR, f"tmp_slurm_{time.time()}.sh")
TMP_BASH_FILE = osp.join(THIS_DIR, f"tmp_bash_{time.time()}.sh")

CKPT_SEPARATOR = "__CKPT_SEP__"


def main(
    ckpt_dir,
    slurm_script,
    prefix,
    jobs_per_gpu,
    min_ckpts,
    partition,
    force,
    tb_name,
    logs_name,
):
    log_dir = get_log_dir(ckpt_dir, logs_name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    if force:
        print(
            "Force flag is set. Deleting all log files that don't have stats."
        )
        for log_file in glob.glob(osp.join(log_dir, "*.log")):
            with open(log_file, "r") as f:
                log_contents = f.read()
            if "Average episode " not in log_contents:
                print(f"Deleting {log_file}")
                os.remove(log_file)
        for i in glob.glob(osp.join(ckpt_dir, f"*.{prefix}_queued")):
            os.remove(i)
        print(f"Deleted all .{prefix}_queued dummy files.")

    if not osp.exists(ckpt_dir):
        print(f"Waiting for {ckpt_dir} to exist...")
        while not osp.exists(ckpt_dir):
            time.sleep(5)

    evaluator = Evaluator(
        ckpt_dir,
        slurm_script,
        jobs_per_gpu,
        prefix,
        partition,
        logs_name,
    )
    stats_log_count = count_log_files(ckpt_dir, logs_name, needs_stats=True)
    first = True
    while first or count_log_files(ckpt_dir, logs_name) < min_ckpts:
        first = False
        evaluator.submit_eval_jobs()
        if min_ckpts != -1:
            time.sleep(60)  # Check once every minute

        # Remove queued files AND their corresponding log file, if the queued
        # file is over 10 hours old AND the log file does not contain stats
        for i in glob.glob(osp.join(ckpt_dir, f"*.{prefix}_queued")):
            if is_file_older_than_n_hours(i, 10):
                os.remove(i)
                log_dir = get_log_dir(ckpt_dir, logs_name)
                log_file = osp.join(
                    log_dir, osp.basename(i).replace(f".{prefix}_queued", ".log")
                )
                if osp.exists(log_file):
                    with open(log_file, "r") as f:
                        log_contents = f.read()
                    if "Average episode " in log_contents:
                        continue
                    os.remove(log_file)
                    print(f"Deleted {i} and its corresponding log file.")
                else:
                    print(f"Deleted {i} but couldn't find its log file.")

        new_stats_log_count = count_log_files(
            ckpt_dir, logs_name, needs_stats=True
        )
        if new_stats_log_count > stats_log_count:
            print(f"Found {new_stats_log_count} stats log files.")
            stats_log_count = new_stats_log_count
            try:
                print("Attempting to update tb logs...")
                logs_to_tb(
                    get_log_dir(ckpt_dir, logs_name),
                    replace=True,
                    tb_name=tb_name,
                )
            except Exception as e:
                print(f"Failed to convert logs to tensorboard: {e}")
                print("Continuing anyway.")


class Evaluator:
    def __init__(
        self,
        ckpt_dir,
        slurm_script,
        jobs_per_gpu,
        prefix,
        partition,
        logs_name,
    ):
        self.ckpt_dir = osp.abspath(ckpt_dir)
        self.slurm_script = slurm_script
        self.jobs_per_gpu = jobs_per_gpu
        self.prefix = prefix
        self.partition = partition
        self.generate_bash_script()
        self.generate_slurm_script()
        self.logs_name = logs_name

    def generate_bash_script(self):
        with open(SINGLE_CKPT_TEMPLATE, "r") as f:
            bash_cmds = f.read()
        bash_cmds += self.extract_habitat_cmd() + "\n"
        with open(TMP_BASH_FILE, "w") as f:
            f.write(bash_cmds)

    def generate_slurm_script(self):
        with open(self.slurm_script, "r") as f:
            slurm_cmds = f.read()
        eval_cmd = f"bash {TMP_BASH_FILE}" + " ${SLURM_CHECKPOINTS}"
        slurm_cmds = slurm_cmds.replace(self.extract_habitat_cmd(), eval_cmd)
        with open(TMP_SLURM_FILE, "w") as f:
            f.write(slurm_cmds)

    def extract_habitat_cmd(self):
        with open(self.slurm_script, "r") as f:
            file_contents = f.read()
        # Join lines that are split with a backslash
        file_contents = file_contents.replace(
            "\\\n", "THERE_ONCE_WAS_A_NEWLINE"
        )
        lines = file_contents.splitlines()
        for line in lines:
            if line.startswith("srun") and "--exp-config" in line:
                line_no_srun = line[len("srun ") :]
                # Remove leading spaces
                while line_no_srun[0] == " ":
                    line_no_srun = line_no_srun[1:]
                return line_no_srun.replace("THERE_ONCE_WAS_A_NEWLINE", "\\\n")

    def submit_eval_job(self, checkpoints):
        """Submit a job to evaluate all given checkpoints. The job will be
        wrapped in a script that will create dummy files for each checkpoint
        before running the slurm script."""
        indices_str = "_".join([c.split(".")[-2] for c in checkpoints])
        job_name = f"{self.prefix}_{indices_str}"
        slurm_out_dir = osp.join(osp.dirname(self.ckpt_dir), "slurm_eval_out")
        os.makedirs(slurm_out_dir, exist_ok=True)
        out_file = osp.join(slurm_out_dir, f"{job_name}.out")
        log_dir = get_log_dir(self.ckpt_dir, self.logs_name)

        sbatch_cmd = [
            "sbatch",
            "--job-name",
            job_name,
            "--output",
            out_file,
            "--error",
            out_file,
            "--open-mode=append",
            "--ntasks-per-node",
            str(len(checkpoints)),
            "--export=ALL,SLURM_CHECKPOINTS="
            f"{CKPT_SEPARATOR.join(checkpoints)},"
            f"SLURM_LOG_DIR={log_dir}",
        ]
        if self.partition is not None:
            sbatch_cmd.extend(["--partition", self.partition])
            if self.partition == "overcap":
                sbatch_cmd.extend(["--account", "overcap"])
        sbatch_cmd.append(TMP_SLURM_FILE)
        print(" ".join(sbatch_cmd))

        # Create the corresponding dummy .queued files
        for ckpt in checkpoints:
            dummy_file = ckpt.replace(".pth", f".{self.prefix}_queued")
            with open(dummy_file, "w") as f:
                f.write("")

        subprocess.check_call(sbatch_cmd, env=os.environ)

    def submit_eval_jobs(self):
        # 1. Get all checkpoints that have not been queued
        checkpoints = self.get_unqueued_checkpoints()
        inds_str = ", ".join([c.split(".")[-2] for c in checkpoints])
        now = str(datetime.now())[:19]
        print(
            f"{now}: Found {len(checkpoints)} checkpoints to evaluate"
            f"{':' if len(checkpoints) > 0 else '.'} {inds_str}"
        )
        # 2. Chunk this flat list into a list of lists, where each sublist has
        # self.jobs_per_gpu elements (except the last one)
        checkpoints = [
            checkpoints[i : i + self.jobs_per_gpu]
            for i in range(0, len(checkpoints), self.jobs_per_gpu)
        ]
        # 3. Submit a job for each sublist
        for idx, ckpts in enumerate(checkpoints):
            print(f"Submitting job {idx + 1} of {len(checkpoints)}:")
            self.submit_eval_job(ckpts)
            print()  # add a newline for neatness

    def get_unqueued_checkpoints(self):
        """Get all checkpoints that neither have a corresponding dummy .queued
        file nor a corresponding log file."""

        def queued_exists(ckpt):
            return osp.exists(ckpt.replace(".pth", f".{self.prefix}_queued"))

        log_dir = get_log_dir(self.ckpt_dir, self.logs_name)

        def log_exists(ckpt):
            basename = osp.basename(ckpt).replace(".pth", ".log")
            return osp.exists(osp.join(log_dir, basename))

        checkpoints = glob.glob(osp.join(self.ckpt_dir, "*ckpt.*.pth"))
        unqueued_checkpoints = [
            c
            for c in checkpoints
            if not queued_exists(c) and not log_exists(c)
        ]
        # Each ckpt has basename "ckpt.N.pth"; sort in descending order.
        unqueued_checkpoints.sort(key=lambda x: -int(x.split(".")[-2]))
        return unqueued_checkpoints


def get_log_dir(ckpt_dir, logs_name):
    """Get the log directory for the given ckpt directory."""
    return osp.join(osp.dirname(osp.abspath(ckpt_dir)), logs_name)


def count_log_files(ckpt_dir, logs_name, needs_stats=False):
    """Count the number of log files in the given directory."""
    log_dir = get_log_dir(ckpt_dir, logs_name)
    logs = glob.glob(osp.join(log_dir, "*.log"))
    if needs_stats:
        filtered_logs = []
        for log in logs:
            with open(log, "r") as f:
                log_contents = f.read()
            if "Average episode " in log_contents:
                filtered_logs.append(log)
        return len(filtered_logs)
    return len(logs)


def is_file_older_than_n_hours(file_path, n=10):
    if not os.path.exists(file_path):
        return False  # File doesn't exist

    creation_time = os.path.getctime(file_path)
    current_time = datetime.now().timestamp()
    time_difference = current_time - creation_time

    hours_difference = time_difference / (60 * 60)  # Convert seconds to hours

    if hours_difference > n:
        return True  # File is older than 10 hours
    else:
        return False  # File is not older than 10 hours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_dir", type=str, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "slurm_script", type=str, help="Path to template slurm script"
    )
    parser.add_argument(
        "prefix", type=str, help="Job name prefix (any str w/o commas)"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="First removes all .queued files and logs files without stats",
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--jobs-per-gpu",
        type=int,
        help="Number of jobs per GPU (default=2)",
        default=2,
    )
    parser.add_argument(
        "-m",
        "--min-ckpts",
        type=int,
        help="Minimum number of ckpts",
        default=-1,
    )
    parser.add_argument(
        "-p", "--partition", help="Slurm partition to submit jobs to"
    )
    parser.add_argument(
        "-t",
        "--tb-name",
        help="Name of the tensorboard log directory (default=tb_eval)",
        default="tb_eval",
    )
    parser.add_argument(
        "-l",
        "--logs-name",
        help="Name of the log directory (default=logs)",
        default="logs",
    )
    args = parser.parse_args()
    main(**vars(args))
