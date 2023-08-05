# assignment-churn-prediction

Small "quick win" data product ready to serve. Some features include:
- Initial data exploration and model baseline in `notebooks/DataExploration_initial_investigation.ipynb`
- Training script with configuration file instructions setup `train_instruction.py`
- Logging training runs to mlflow (locally) train and inspect with `mlflow server --backend-store-uri ./mlflow`
- Setup `setup.py` to easily share model code and implementations for inference 
- FastAPI endpoint serving the trained model (see below how to start in docker)


# Start FastAPI
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
Based on user interaction with the product, identify users likelihood to churn based on historical data of previously churned users.
The available data is suited to train a classifier to predict the churn state of a user. 
At the same time the classifier produces a probability with which the user is predicted to be in the churned state.
Note that the fact that the model is supposed to predict the churn state (and probability) of users in the future, special care needs to be taken for the features that include (absolute) time information.

## What is your overall approach?
Produce a data solution as quickly as possible that is suitable for systematic improvements.
By not wasting time to train a(ny) model that generates reasonable predictions and deploy the model, we can kick off the feedback loop as quick as possible.
In addition, we do not waste time on making wild guesses on "best features" to engineer before we are able to efficiently verify our reasoning and measure the impact of such additional features.
The next steps would include:
- Gather more information about the business case to make informed decision for improvements
- Analyze the models short-comings by analyzing misclassified cases, the feature importance of similar approaches, to formulate and implement necessary improvements to bump the performance of the model.
- Potentially, clean up the setup in preparation of continuous development and improvement.

## What could be user retention strategies?
Could be as easy as sending a reminder to the user to check on recent developments. 
For such low-cost retention strategies a focus on the models recall is likely most beneficial.

## How did you choose the specific model?
Without any deep considerations! The goal for the initial implementation should be to produce a working model as fast as possible.
I choose a random forest, which are a good starting point as they are robust against the features scaling, generalize well and are usually fast to train.
However, it could have equally been gradient boosting, logistic regression, support vector machines, ... you name it.

## How did you choose the evaluation metric?
The training script performs a hyperparameter optimization based on the F1 score. 
This seems to be a good generic starting point, striking a happy medium between recall and precision.
In practice, I like to tailor the performance score to the business problem at hand, such as how easy to implement the retention strategies are.
However, in absence of sufficient information, I don't like to take wild guesses.
