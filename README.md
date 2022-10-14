# Stock Market Prediction
Containerized level code for Stock Market Prediction

## **Summary**
Solving the problem of predicting the direction of the stock market is every trader's dream.
For being able to predict the stock market comes with the added benefit of knowing exactly when to buy or sell a stock.

This project attempts to assist in this prediction task by utilizing a particular type of deep neural network known as Long-Short term memory or LSTM for short.
I implement a state-of-the-art solution for hyper-parameter optimization. 
The process is known as Bayesian optimization, it uses a gaussian process to try and approximate the hyper-parameter space with respect to a metric you provide.

The metric I used is F-beta score, with a beta of 0.125, which weighs heavier towards precision by a good amount.
This is a good number for our use case as I am assuming a conservative investment style in which you prioritize a correct prediction for the positive case.

The results are decent, we are able to get a precision rate of ~60%

## **Important Files**

Some important files and their descriptions are as follows:
 - `src/data/data_loader.py`
   - This file contains the class that I utilize to get the data from yahoo finance
   - It is also in charge of keeping the data up to date
 - `src/features/features_builder.py`
   - This file contains a class for manipulating the data we have into digestible arrays for training modeling and predicting.
   - It is also responsible for providing the ready-to-use data
 - `src/models/model_manager.py`
   - This file contains a class for managing the various models on each particular stock ticker
   - This includes training, validating, loading, saving, and predicting
 - `app.py`
   - This file contains the main logic for the streamlit rendering
   - Also in charge of control flow of my application

Access my application: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://rubenren-stockmarket-datascience-app-mlj36e.streamlitapp.com/)

Spin up your own copy using my code:
1. Without Docker (if this does not work use the Docker method)
    1. Setup a virtual environment and activate it
    2. Clone my repository
    3. Go to the root directory of the project and run `pip3 install -r requirements`
    4. Run `streamlit run app.py`
    5. Your default web browser should open a page with my app running from it, or you can connect to it using the ips they gave you in the console
    6. Provide your inputs to the UI and have fun!
2. With Docker
   1. Have Docker installed and Docker-Compose installed on your system, ensure you can run them from the command line
   2. Start Docker
   3. Clone my repository
   4. Go to the root directory of the project and run `docker compose up` or `docker-compose up`
   5. Go to a web browser and type `http://localhost:8501` into the address bar
   6. Provide your inputs to the UI and have fun!


How to use my Application:
1. You will see an interface that asks for a stock ticker at the top. This is used to add a ticker symbol to our tracking file.
   1. If you have a specific stock in mind, try adding it and see if we can download the data
2. The next part on the application is a selection box with a bunch of different stocks to choose from.
   1. There are a few popular ones that have been added to tracking
3. You should notice the train, predict, and evaluate buttons right below the selection box
   1. Train will overwrite any current model that has been trained on the stock, and train a new classifier over it.
   2. Predict will either load a model from its data or train a new model on-the-fly and make use of that one. Once it has a model it can use, it will run the prediction for the last day in its database.
   3. For every prediction there is a sentence telling you what the model believes, and the number associated with its guess, as well as the date the prediction is using as a reference (i.e. could technically be a 4-day prediction)
   4. The evaluate button is used to evaluate the model that is currently loaded, it will return a list of metrics that the model accomplished on the recent 20% of its history
   5. The precision and recall are incredibly important for this screen, as precision is what we are optimizing for, and recall will let you know if your model is just guessing all ups or all downs (good precision should be above .50 and good recall should be below 1.0)
4. The last thing to notice is the hyper-parameter optimization option and the number of epochs box
   1. The hyper-parameter optimization option allows you to train using the Bayesian Optimization, this will definitely take a while to run (so go grab a snack while it runs!)
   2. All that is needed to use this feature is to check that box and click train
   3. The number of epochs number selector is used for determining the max number of epochs the system will run for each training iteration (hyper-parameter optimization will run 40 training sessions before giving you a model)
5. Have fun predicting!