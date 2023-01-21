import argparse
import gzip
import json
import os
import os.path as osp

import tqdm

print("Importing habitat...")
from habitat_baselines.config.default import get_config

print("...done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_config", type=str, help="Path to the config file")
    parser.add_argument("split", type=str, help="Which split to look through")
    parser.add_argument(
        "episode_ids", type=int, nargs="+", help="Episode ids to extract"
    )
    parser.add_argument(
        "-o", "--output-split", type=str, help="Name of new split", default="debug"
    )

    args = parser.parse_args()

    config = get_config(args.exp_config)
    data_path_template = config.habitat.dataset.data_path
    data_path = data_path_template.format(split=args.split)
    # Use os.walk on data_path to find all the json.gz files
    gz_files = []
    for root, dirs, files in os.walk(osp.dirname(data_path)):
        for file in files:
            if file.endswith(".json.gz"):
                gz_files.append(osp.join(root, file))
    filtered_episodes = []
    for gz_file in tqdm.tqdm(gz_files):
        with gzip.open(gz_file, "rt") as f:
            data_dict = json.load(f)
            filtered_episodes.extend(
                [
                    ep
                    for ep in data_dict["episodes"]
                    if int(ep["episode_id"]) in args.episode_ids
                ]
            )
    new_data_path = f"data/{args.output_split}.json.gz"
    os.makedirs("data", exist_ok=True)
    with gzip.open(new_data_path, "wt") as f:
        json.dump({"episodes": filtered_episodes}, f)

    print("The following command opt will work with the new split:")
    print(f"habitat.dataset.data_path='data/{args.output_split}.json.gz'")


if __name__ == "__main__":
    main()
