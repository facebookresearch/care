# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from functools import partial
import requests
import pandas as pd
import os
from typing import Dict, List
import multiprocessing
import utils
import argparse

# Metadata parameters to save when downloading the post metadata
parameters_to_save = [
    "id",
    "num_comments",
    "is_original_content",
    "parent_id",
    "link_id",
    "subreddit",
    "permalink",
    "subreddit_type",
    "category",
    "url",
    "submission-type",
    "lang",
    "title",
    "selftext",
    "header_title",
    "submit_text",
    "metadata",
]

# This function uses pushshift.io to download all metadata the posts in the CARE database. data_file should point to a csv containing the post ids in the CARE database. The parameter params_to_keep enumerates the parameters to save. Increase cpus_to_use if for more multiprocessing.
def download_all_sub_data(
    sub_ids: List[str] = None,
    data_file: str = None,
    cpus_to_use: int = 2,
    n: int = 10,
    output_file: str = None,
    chunked_folder: str = None,
    params_to_keep: List[str] = utils.parameters_to_save,
) -> None:
    if data_file is None:
        data_file = "./care_db_ids_and_labels.csv"
    if sub_ids is None:
        assert os.path.exists(data_file)
        sub_ids_df = pd.read_csv(data_file, sep="\t")
        sub_ids = [x for x in list(sub_ids_df["id"]) if isinstance(x, str)]
    pool = multiprocessing.Pool(cpus_to_use)
    chunked_list = sorted([sub_ids[i : i + n] for i in range(0, len(sub_ids), n)])
    func = partial(
        download_sub_data_one_chunk,
        output_file_path=chunked_folder,
        chunked_list=chunked_list,
        params_to_keep=params_to_keep,
    )
    pool.map(func, range(len(chunked_list)))
    aggregate_chunks(output_file=output_file)
    pool.close()
    pool.join()


# Helper function for download_all_sub_data. By defaults it saves to care/data/chunks/post_id_metadata_{index}.json
def download_sub_data_one_chunk(
    index: int,
    chunked_list: List[List[str]],
    attempt: int = 1,
    output_file_path: str = None,
    params_to_keep: List[str] = None,
) -> bool:
    sub_ids = chunked_list[index]

    if output_file_path is None:
        output_file_path = f"./data/chunks/post_id_metadata_{index}.json"

    if os.path.exists(output_file_path):
        return True

    if not os.path.exists(os.path.dirname(os.path.abspath(output_file_path))):
        os.makedirs(os.path.dirname(os.path.abspath(output_file_path)))

    if attempt == 5:
        return False
    try:
        response = requests.get(
            "https://api.pushshift.io/reddit/submission/search?ids=" + ",".join(sub_ids)
        )
        data = response.json()["data"]

        if params_to_keep is not None:
            filtered_data = []
            for entry in data:
                new_entry = {}
                for param in params_to_keep:
                    if param in entry:
                        new_entry[param] = entry[param]
                filtered_data.append(new_entry)
            data = filtered_data

        with open(f"{output_file_path}", "w", encoding="utf8") as fh:
            fh.write(json.dumps(data) + "\n")
        return True
    except:
        download_sub_data_one_chunk(
            index, chunked_list, attempt=attempt + 1, output_file_path=output_file_path
        )


# Aggregates all the downloads into one file. By default, it saves to care/data/post_id_metadata.json
def aggregate_chunks(
    output_file_path: str = None, chunked_output_folder: str = None
) -> None:
    if output_file_path is None:
        output_file_path = f"./data/post_id_metadata.json"
    if chunked_output_folder is None:
        chunked_output_folder = f"./data/chunks/"

    all_data = []

    for file in os.listdir(chunked_output_folder):
        with open(os.path.join(chunked_output_folder, file), "r") as fin:
            data = json.load(fin)
        all_data.extend(data)

    with open(f"{output_file_path}", "w", encoding="utf8") as fh:
        for example in all_data:
            fh.write(json.dumps(example) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpus",
        type=int,
        required=False,
        default=2,
        help=f"Number of cpus to use for multiprocessing.",
    )

    parser.add_argument(
        "--n",
        type=int,
        required=False,
        default=10,
        help=f"Number of post ids for each job.",
    )

    parser.add_argument(
        "--data_file", type=str, default=None, help="Path the to csv with post ids."
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Write the metadata to this file."
    )

    parser.add_argument(
        "--chunk_dir",
        type=str,
        default=None,
        help="Write the batch metadata to this directory. This can be deleted after aggregation.",
    )
    args = parser.parse_args()

    download_all_sub_data(
        data_file=args.data_file,
        cpus_to_use=args.cpus,
        n=args.n,
        output_file=args.output_file,
        chunked_folder=args.chunk_dir,
    )
