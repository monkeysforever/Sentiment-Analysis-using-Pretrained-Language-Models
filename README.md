# Introduction
The repository is intended to be used for sentiment analysis but can be extender to text classification tasks. BERT is used on the IMDB reviews polarity dataset but can be used with any dataset in csv format with columns namely, text and label.

# Requirements
Run the following command to install required dependencies
```
pip install requirements.txt
```

# Data
Run the following command to download the IMDB dataset and generate train and test csv files. Train and test csv files have 25000 sentences each.
```
python create_dataset.py
```

# Example 
Below is an example for running the classifier, after generating the data. GPU is required.
```
python train.py --epochs 10 
```

# Results

|Model  |Accuracy|
|-------|--------|
| BERT  |  85.32 |
