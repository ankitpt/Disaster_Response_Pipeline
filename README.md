# Disaster Response Pipeline Project

### Table of Contents

a. [Project Motivation](#motivation)
b. [File Descriptions](#datadescriptions)
c. [Instructions](#instructions)
d. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
The motivation behind executing this project is to put data engineering skills into practice ie implementing an ETL pipeline followed by a machine learning pipeline to classify disaster messages provided by [Figure 8](https://appen.com/) now acquired by Appen.

## File Descriptions<a name="filedescriptions"></a>

There are three main folders:
1. data
    - disaster_categories.csv: dataset including all the categories 
    - disaster_messages.csv: dataset including all the messages in original language and trasnlated to English
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web applicatin

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
As mentioned, the data was obtained from [Figure 8](https://appen.com/). Further, I would also like to acknowledge Udacity Data Scientist Nanodegree instructors for their efforts towards making the students understand various steps of an ETL and ML pipeline.