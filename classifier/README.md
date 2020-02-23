# Classifier

This directory contains a script for training, testing, and running a classifier on any input data.

Following is the usage for the classifier:
usage: classifier.py [-h] [--model_name MODEL_NAME]
                          [--pkl_dir PKL_DIR] [--data_path DATA_PATH]
                          [--text_field TEXT_FIELD]
                          [--delimiter DELIMITER]
                          [--result_path RESULT_PATH] (-r | -e | -a)

Train model or classify tweets

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name for model, to save (train) or load (test/tag)
  --pkl_dir PKL_DIR     Path to directory to save pickle of model and other
                        files needed to transform new data such that they can
                        be classified.
  --data_path DATA_PATH
                        Path to data, must be CSV file
  --text_field TEXT_FIELD
                        Name of field in data file in which to find the text
                        data
  --delimiter DELIMITER
                        Delimiter of data file
  --result_path RESULT_PATH
                        Path to save result; if not provided, results will be
                        printed to stdout
  -r, --train
  -e, --test
  -a, --tag

The progress of the classifier will be output in the shell. If no result path is provided, each tweet and its predicted category will be printed to shell as well.
