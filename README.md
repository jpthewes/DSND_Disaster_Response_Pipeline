# Disaster Response Pipeline Project

### Project Overview:
This project aims to classify messages from disasters which appear on social networks or get sent to crisis management into categories. This classification helps to direct the right responders to the right people. 
For this a dataset from FigureEight is used and Data Engineering skills are applied. For further steps please refer to the Details section.

### Instructions:
1. The code should be run using Python3 with all the necessary dependencies installed.

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Details:
```
├── app							# webapp
│   ├── run.py					# flask file that runs the webapp
│   └── templates
│       ├── go.html				# classification results page
│       └── master.html			# webapp main page
├── data
│   ├── disaster_categories.csv	# processed training data
│   ├── disaster_messages.csv	# processed training data
│   ├── DisasterResponse.db		# database with cleaned data
│   └── process_data.py			# script to run the ETL process
├── models
│   ├── classifier.pkl			# classifier model used in webapp
│   └── train_classifier.py		# trains the model with the data
└── README.md
```