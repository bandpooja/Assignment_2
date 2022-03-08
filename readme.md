# Assignment 2

For `Assignment-2` I have used the **Brown** corpus to build *N-gram* language model. For testing I used **Brikbeck** corpus.

The task aims to suggest auto completion for sentence.

## Setup

To setup the environment which I used to create this project. Just run 

```
pip install -r requirements.txt

```

## Structure

All the essential/useful functions have been added to the **utils folder** and the python script `assignment2.py` and the notebook `assignment2.ipynb` just uses these function to generate results.

## Parallelization 

Since the task to be performed is too intensive for one machine we need to parallelize the job distribution in a machine and across different machines as well. 


To parallelize the tasks across different cores a CPU, I used `concurency.futures` **ProcessPool** because the tasks are CPU intensive. (I also used Threading Pool but the process pool gave a better speed up so I used that finally). Whereas to parallelize the runs across multiple machines I used caching the function results to local-disk to avoide re-computing the same detail.
