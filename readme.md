# Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting with pseudolabeling
- This is a repo forked from G2P2: https://github.com/WenZhihao666/G2P2
- Please visit the original repo and its paper before reading this README
- This branch provides extension by using pseudo labeling on unlabeled nodes - for zero-shot training only


# General ideas:
- Since the labels for classification are also in text format, we can use the text encoder to extract text embeddings
- Compare the graph embeddings of unlabeled nodes with labels embeddings to get pseudo label
- Using these pseudo label as samples when running zero-shot classification

# New addition:
- New params:
	- num_sample: number of pseudo labeled nodes to use for training
	- conf: confidence score filter for pseudo labeled nodes
- New files:
	- main_test_pseudo_label.py: pseudo labeling for all labels - DOES NOT HAVE CONFIDENCE FILTER
	- main_test_pseudo_label_filter_conf.py: pseudo labeling for all labels - WITH CONFIDENCE FILTER
	- main_test_pseudo_label_task_split.py: in a N-task setting, only apply pseudo labeling for labels within N tasks set, not for all tasks - WITH CONFIDENCE FILTER
	- auto_run files: script to run experiments with various hyperparameters
