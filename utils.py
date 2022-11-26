# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

# Clustering into seven affective responses.
CLUSTER_MAP = {
    "disgusted": "angered",
    "saddened": "saddened",
    "amused": "amused",
    "angered": "angered",
    "disappointed": "saddened",
    "interested": "amused",
    "impressed": "approving",
    "excited": "excited",
    "inspired": "approving",
    "annoyed": "angered",
    "admiring": "approving",
    "scared": "scared",
    "worried": "scared",
    "anxious": "scared",
    "adoring": "adoring",
    "approving": "approving",
    "attracted": "adoring",
    "entertained": "amused",
}

CORE_AFFECTS = [
    "adoring",
    "angered",
    "amused",
    "approving",
    "excited",
    "saddened",
    "scared",
]

# This function is for clustering according to the hierarchy defined in CLUSTER_MAP and/or filtering for the affects defined in CORE_AFFECTS.
def cluster_and_filter(
    affect_map: Dict[str, int], cluster: bool = True, filter: bool = True
) -> Dict[str, int]:
    new_affect_map = {}

    for orig_k, orig_v in affect_map.items():
        if not cluster or orig_k not in CLUSTER_MAP:
            k = orig_k
        else:
            k = CLUSTER_MAP[orig_k]

        if filter and k not in CORE_AFFECTS:
            continue

        if k not in new_affect_map:
            new_affect_map[k] = 0

        new_affect_map[k] += orig_v

    return new_affect_map
