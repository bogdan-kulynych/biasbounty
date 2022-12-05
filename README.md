# Bias Bounty Submission

## Setup Instructions

### Data
Unpack the zip file with data to the `data/` folder.

### Python Dependencies
The code requires Python 3.8+. Install Python packages with `pip install -r requirements.txt`

## Running

### Evaluation of Score and Generation of Submission File
Run `python src/eval.py --model_path models_submission`

### Training the models
To train the models, run `python src/train.py`

## About the Submission
We tried to start with a strong baseline of the pretrained Facenet, and fine-tuned it on the
provided data using SGD with some hyperparameter search. The initial idea was to add importance
sampling coupled with the gradient noise regularizer (https://arxiv.org/abs/2204.03230), but we
did not manage to get to that.
