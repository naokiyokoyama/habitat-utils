import argparse
import gzip
import random

import pathos.multiprocessing as multiprocessing
import os
import os.path as osp

import habitat
import tqdm
import yaml
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from omegaconf import DictConfig


parent_dir = osp.dirname(osp.abspath(__file__))
with open(osp.join(parent_dir, "train_val_splits.yaml"), "r") as f:
    TRAIN_VAL_SPLITS = yaml.safe_load(f)

def _generate_fn(
    scene: str,
    scene_dir: str,
    cfg: DictConfig,
    out_dir: str,
    num_episodes_per_scene: int,
    is_hm3d: bool = False,
    split="train",
):
    if is_hm3d:
        scene_name = scene.split("-")[-1]
        scene_path = f"hm3d/{split}/{scene}/{scene_name}.basis.glb"
    else:
        scene_name = scene
        scene_path = f"gibson/{scene}.glb"

    # Skip this scene if a dataset was or is being generated for it
    out_file = osp.join(out_dir, f"{split}/content/{scene_name}.json.gz")
    if osp.exists(out_file) or osp.exists(out_file + ".incomplete"):
        return
    # Create an empty file so other processes know this scene is being processed
    with open(out_file + ".incomplete", "w") as f:
        f.write("")

    # Insert path to scene into config so it gets loaded
    full_scene_path = osp.join(scene_dir, scene_path)
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = full_scene_path
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim,
            num_episodes_per_scene,
            is_gen_shortest_path=False,
        )
    )

    for ep in dset.episodes:
        ep.scene_id = scene_path

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    os.remove(out_file + ".incomplete")


def generate_dataset(
    config_path: str,
    scene_dir: str,
    split: str,
    out_dir: str,
    dataset_type: str,
    overrides: list,
    num_episodes_per_scene: int,
):
    cfg = habitat.get_config(config_path=config_path, overrides=overrides)
    is_hm3d = dataset_type == "hm3d"
    scenes = TRAIN_VAL_SPLITS[dataset_type][split]
    out_file = osp.join(out_dir, f"{split}/{split}.json.gz")
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write('{"episodes": []}')

    _generate_fn_partial = lambda x: _generate_fn(
        x,
        scene_dir,
        cfg,
        out_dir,
        num_episodes_per_scene,
        is_hm3d,
        split,
    )
    # Shuffle order of elements in scenes
    random.shuffle(scenes)
    with multiprocessing.Pool(27) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn_partial, scenes):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-config",
        required=True,
        help="Path to config yaml used to setup the simulator",
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Dataset type to generate. One of [hm3d, gibson]",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Which dataset split to generate. One of [train, val]",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Path to output directory for dataset",
    )
    parser.add_argument(
        "-s",
        "--scenes-dir",
        help="Path to the scene directory",
        default="data/scene_datasets",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Modify config options from command line",
    )
    parser.add_argument(
        "-n",
        "--num_episodes_per_scene",
        type=int,
        help="Number of episodes per scene",
        default=1e3,
    )
    args = parser.parse_args()
    assert args.dataset_type in [
        "hm3d",
        "gibson",
    ], f"Invalid dataset type {args.dataset_type}"
    generate_dataset(
        args.exp_config,
        args.scenes_dir,
        args.split,
        args.out_dir,
        args.dataset_type,
        args.overrides,
        args.num_episodes_per_scene,
    )