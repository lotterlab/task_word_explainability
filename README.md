# task_word_explainability

Code for: S Agarwal, YR Semenov, W Lotter. Representing visual classification as a linear combination of words. _Proceedings of the 3rd Machine Learning for Health symposium, PMLR_ (2023).


![methods_fig](https://github.com/lotterlab/task_word_explainability/assets/5847322/fc43075a-e1fd-4171-a659-c66c7e4f8fc7)


## Set-up/Requirements
1. Install CLIP and its dependencies by following instructions at: https://github.com/openai/CLIP
2. Install packages in `requirements.txt` in order to run all scripts. The code should generally be version agnostic for these packages.

## Usage
`fit_words.py`: Illustrates fitting a linear classifier based on CLIP image embeddings and subquently estimating the classifier based on a linear combination of word embeddings.

`plotting.py`: Plots the regression word weights.
