# The CARE Dataset for Affective Response Detection

The CARE Dataset is described in the following paper: [[https://arxiv.org/abs/2209.13331](https://arxiv.org/abs/2209.13331)]

```
@article{DBLP:journals/corr/abs-2201-11895,
  author    = {Jane Dwivedi-Yu and
               Alon Y. Halevy},
  title     = {The CARE Dataset for Affective Response Detection},
  journal   = {CoRR},
  volume    = {abs/2201.11895},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.11895},
  eprinttype = {arXiv},
  eprint    = {2201.11895},
  timestamp = {Wed, 02 Feb 2022 15:00:01 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-11895.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Environment

    conda create -n care -y python=3.7 && conda activate care
    pip: -r requirements.txt

## Downloading the dataset

Download the post ids and care labels with:
    wget dl.fbaipublicfiles.com/care/care_db_ids_and_labels.csv

To download the metadata associated with the post ids, run 

    python download_posts.py
   
By default, it uses the following defaults:

    python download_posts.py --data_file ./care_db_ids_and_labels.csv --output_file ./data/post_id_metadata.json --chunk_dir ./data/chunks/ --n 10 --cpus 2
  
This process all post ids from ./data/care_db_ids_and_labels.csv, use 2 cpus to process batches of 10 post ids at a time, write the intermediate outputs to ./data/chunks/ and finally aggregate them into the file ./data/care_db_ids_and_labels.csv

## Running the CARE method

The CARE method can be run to label posts given the comment texts. See ```care_predict.py``` for an example.

## Running CARE-BERT

See ```care_bert.py``` for an example of how to load and use CARE-BERT for inference. The model with the command: 

    wget dl.fbaipublicfiles.com/care/care_bert.pth

## Licensing

All code and data is released under CC-BY-NC licensing. See our LICENSE file for licensing details.
