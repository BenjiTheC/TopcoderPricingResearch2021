# Topcoder Pricing Research 2021

> This is the repo for data analysis and model building of Topcoder Pricing reserach by Benjamin.
> Reason for creating a new repo:
>
> 1. Previous repos are messed up with unorginized code and notebook.
> 2. I'm switching from TensorFlow2 to PyTorch.
> 3. It's refreshing 🤪

## Dependency

This repo is written in Python 3.9.1

run following command under the repo directory _after installation of Python 3.9._

```sh
python3 -m venv venv
pip3 install -r requirements.txt
```

## Feature Engineering

The training feature matrix will be composed of multiple parts:

1. Word2Vec vectors from challenge description text
2. Metadata such as challenge duration, project which owns the challenge
3. Encoded array of challenge tags, singles, 2, 3 and 4 combinatioon
4. Competing challenges that are in the market

### Tags

Rules for manufacturing feature from challenge tags:

1. Count the number of tags, number of 2-tag combination, number of 3-tag combination, number of 4-tag combination
2. Pick the top 25 tag/combinations of each group.
3. Compute the log value of counts, convert it to 4 softmax function
4. Calculate summary of softmax score of 4 group of tag combination, as 4 feature digit array
5. Calculate the binary encoded array of tag combinations as a 100 feature digit array
