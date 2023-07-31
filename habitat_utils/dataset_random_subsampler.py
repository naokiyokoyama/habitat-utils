import argparse
import glob
import gzip
import json
import os
import random
from typing import List


def random_episodes(json_gz_file_path: str, N: int) -> str:
    """
    Randomly selects N episodes from a JSON.gz file, creates a new JSON.gz file
    with only the selected episodes, and returns the path to the new file.

    Parameters:
        json_gz_file_path (str): Path to the input JSON.gz file.
        N (int): Number of episodes to select randomly.

    Returns:
        str: Path to the new JSON.gz file containing the selected episodes.
    """
    # Step 1: Read the input JSON data from the gzipped file
    with gzip.open(json_gz_file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the 'episodes' list from the dictionary
    episodes = data.get("episodes", [])

    # Step 2: Randomly select N episodes from the 'episodes' list
    if len(episodes) > N:
        selected_episodes = random.sample(episodes, N)
    else:
        selected_episodes = episodes

    # Create the 'subsampled' directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(json_gz_file_path), "subsampled")
    os.makedirs(output_dir, exist_ok=True)

    # Form the new data dictionary with only the selected episodes
    data["episodes"] = selected_episodes

    # Step 3: Save the resulting data with N episodes into a new JSON.gz file in the
    # 'subsampled' directory
    output_file_path = os.path.join(output_dir, os.path.basename(json_gz_file_path))
    with gzip.open(output_file_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    return output_file_path


def distribute_N_values(X: int, num_bins: int) -> List[int]:
    """
    Distributes a total number of episodes (X) among a given number of bins (num_bins)
    as evenly as possible while still summing up to X.

    Parameters:
        X (int): Total number of episodes desired after distribution.
        num_bins (int): Number of bins (e.g., files) to distribute the episodes.

    Returns:
        List[int]: A list of integers representing the number of episodes to be
                   placed in each bin.
    """
    # Calculate the quotient and remainder of X divided by num_bins
    quotient, remainder = divmod(X, num_bins)

    # Initialize the list with the quotient for each bin
    N_values = [quotient] * num_bins

    # Distribute the remainder among the first few bins
    for i in range(remainder):
        N_values[i] += 1

    return N_values


def process_directory(directory_path: str, X: int) -> None:
    """
    Processes all .json.gz files within the given directory, randomly selecting
    episodes for each file, and saves the new files with the desired total number
    of episodes (X).

    Parameters:
        directory_path (str): Path to the directory containing .json.gz files.
        X (int): Total number of episodes desired after processing.

    Returns:
        None
    """
    # Step 1: Find all .json.gz files within the given directory
    json_files = glob.glob(os.path.join(directory_path, "*.json.gz"))

    if not json_files:
        print("No .json.gz files found in the directory.")
        return

    num_files = len(json_files)

    # Step 2: Distribute N values for each file
    N_values = distribute_N_values(X, num_files)

    for file_path, N in zip(json_files, N_values):
        # Step 3: Use the 'random_episodes' function for each file
        output_file_path = random_episodes(file_path, N)
        print(f"Successfully saved {N} random episodes to {output_file_path}")


def main():
    """
    Main function that parses command-line arguments, and initiates the processing
    of .json.gz files in the specified directory to meet the desired total number
    of episodes (X).
    """
    parser = argparse.ArgumentParser(
        description="Randomly remove episodes from JSON.gz files in a directory."
    )
    parser.add_argument(
        "directory", help="Path to the directory containing .json.gz files."
    )
    parser.add_argument(
        "num_episodes", type=int, help="Total number of episodes desired after processing."
    )
    args = parser.parse_args()

    directory_path = args.directory
    num_episodes = args.num_episodes

    process_directory(directory_path, num_episodes)


if __name__ == "__main__":
    main()
