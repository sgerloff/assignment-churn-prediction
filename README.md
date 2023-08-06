# assignment-churn-prediction

Small "quick win" data product ready to serve. Some features include:
- Initial data exploration and model baseline in `notebooks/DataExploration_initial_investigation.ipynb`
- Training script `train_instruction.py`; configurations are used to control the training parameters, e.g. `instruction/default_instruction.yaml`
- Logging training runs to mlflow (locally) train and inspect with `mlflow server --backend-store-uri ./mlflow`
- Setup `setup.py` to easily share model code and implementations for inference 
- FastAPI endpoint serving the trained model (see below how to start in docker)

## Improvement Baseline Model

In the baseline model, I have dropped the date columns all together to avoid spilling absolute time information to the model.
The first improvement introduced by the training script is to compute new features from the date columns, which express the relative time between the date and the date of last visit.
Introducing this feature, as well as performing a hyperparameter search, the test score improved from `~78%` to `~85%`.

### Train Scores
```text
              precision    recall  f1-score   support

           0       0.83      0.89      0.86     20011
           1       0.88      0.81      0.85     19989

    accuracy                           0.85     40000
   macro avg       0.86      0.85      0.85     40000
weighted avg       0.86      0.85      0.85     40000
```
### Test Scores
```text
              precision    recall  f1-score   support

           0       0.82      0.90      0.86      4989
           1       0.89      0.80      0.84      5011

    accuracy                           0.85     10000
   macro avg       0.85      0.85      0.85     10000
weighted avg       0.85      0.85      0.85     10000
```



# Start Churn Prediction (FastAPI Microservice)
```commandline
docker-compose up
```
Test default endpoint using:
```commandline
scripts/test_fastapi_app.sh
```
(Note: You may need to set the proper permission to execute the script `chmod a+x scripts/test_fastapi_app.sh`)

# FAQ:
## What is your problem understanding?
Based on the users interaction with the product, identify a users likelihood to churn based on historical data of previously churned users.
The available data is suited to train a classifier to predict the "churned" state of a user. 
At the same time the classifier produces a probability with which the user is predicted to be in the churned state. 
Prevention strategies can be employed based on the probability for the user to be in the churned state.

Note that the fact that the model is supposed to predict the churned state (and probability) of users in the future, special care needs to be taken for the features that include (absolute) time information.

## What is your overall approach?
Produce a data solution that is suitable for systematic improvements as quick as possible.
By not wasting time to train a baseline model, we can kick off the feedback loop early in the development process.
In addition, we do not waste time on making wild guesses on "best features" to engineer before we are able to efficiently verify our reasoning and measure the impact of such additional features.
The next steps would include:
- Gather more information about the business case to make informed decision for future improvements
- Systematically improve the models performance by analyzing the current models shortcoming, using e.g. misclassified test cases or feature importance to name a few.
- Potentially, clean up the setup in preparation of continuous development and improvement.

## What could be user retention strategies?
Could be as easy as sending a reminder to the user to check on recent developments. 
In fact, a model that predicts the success probability of retention strategies for the user might be a nice supplementation of the churn prediction model.

## How did you choose the specific model?
Without any deep considerations! The goal for the initial implementation should be to produce a working model as fast as possible.
Based on personal preference, I choose a random forest. 
I consider them a good starting point as they are robust against the features scale, generalize well and are usually fast to train.
However, it could have been any other classification model you prefer (gradient boosting, logistic regression, support vector machines, ... you name it)

## How did you choose the evaluation metric?
The training script performs a hyperparameter optimization based on the F1 score. 
This seems to be a good generic starting point, striking a happy medium between recall and precision.
In practice, I like to tailor the performance score to the business problem at hand, such as how easy to implement the retention strategies are.
However, in absence of sufficient information, I don't like to take wild guesses.
