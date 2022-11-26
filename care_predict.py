# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import regex_pipeline
from typing import Dict, List
from collections import Counter
import pandas as pd
import utils

# Labels posts based on if at least t comments are labeled with the same affect.
def label_posts(
    post_id_to_comment_texts: Dict[str, List[str]], t: int = 5
) -> pd.DataFrame:
    outputs = []

    for post_id, comment_texts in post_id_to_comment_texts.items():
        affects = []
        for comment_text in comment_texts:
            comment_affects = regex_pipeline.get_regex_match_all(comment_text)
            affects.extend(comment_affects)
        affect_map = dict(Counter(affects))
        filtered_affect_map = {}
        for k, v in utils.cluster_and_filter(affect_map).items():
            if v >= t:
                filtered_affect_map[k] = v
        if len(filtered_affect_map) > 0:
            outputs.append([post_id, filtered_affect_map])
    return pd.DataFrame(outputs, columns=["post_id", "affect_map"])


if __name__ == "__main__":
    example_dict = {
        "1": ["This is so funny!!", "Cannot stop laughing at this.", "So hilarious"]
    }
    print(label_posts(example_dict, t=3))
