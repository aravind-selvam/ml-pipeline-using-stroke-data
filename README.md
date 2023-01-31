# 🫀 Heart Stroke Prediction

### Problem Statement and Solution Proposed 

This project aims to solve the problem of Healthcare clinics where they can predict if a patient is likely to get heart stroke based on diagnostic report, the model is done Using Sklearn's supervised machine learning techniques. It is a Classification problem and training are carried out on dataset of previous patients with their diagnostic report with age, gender and history of other disease. Several classification techniques have been studied, the model has been finalized with Random forest and K-Nearest Neighbors in pipeline.

For Detailed EDA and Feature engineering Check out notebook directory. dataset is stored locally inside notebooks directory.

Their performances were compared in order to determine which one works best with our dataset and used them to predict if patient will get heart stroke or not from user input from Flask application.

## 👨‍💻 Tech Stack Used

1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## 🌐 Infrastructure Required.
1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions
5. Terraform

## 💾 Features in the dataset
    - id : a unique identifier that distinguishes each data [int]
    - Gender: Patient's gender ('Male', 'Female', and 'Other') [str]
    - age : Age of the patient [int]
    - Hypertension: Hypertension or high blood pressure is a disease that puts a person at risk for stroke. 0 if the patient does not have hypertension, 1 if the patient has hypertension. [int]
    - heart_disease: Heart disease is a disease that puts a person at risk for stroke. 0 if the patient does not have heart disease, 1 if the patient has heart disease. [int]
    - ever_married : Describes whether the patient is married or not ('Yes' or 'No') [str]
    - work_type : Type of employment or status ('children' for children, 'Govt_job' for civil servants, 'Never_worked' for those who have never worked, 'Private' or 'Self-employed' for entrepreneurs or freelancers) [str]
    - Residence_type : Condition of residence ('Rural' for rural areas and 'Urban' for urban areas) [str]
    - avg_glucose_level : Average amount of glucose (sugar) in the blood [float]
    - bmi : Body Mass Index to measure the stability of body weight with height. [float]
    - smoking_status : Description of smoking ('formerly smoked' for those who have smoked, 'never smoked' for those who have never smoked, 'smokes' for those who smoke, and 'unknown' for those whose smoking status is unknown) [str]


## Project Folder Structure 
```
root/
└── heart_stroke/
    ├── cloud_storage/
    │   ├── __init__.py
    │   └── aws_storage.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── data_validation.py
    │   ├── model_evaluation.py
    │   ├── model_pusher.py
    │   └── model_trainer.py
    ├── configuration/
    │   ├── __init__.py
    │   ├── aws_connection.py
    │   └── mongo_db_connection.py
    ├── constant/
    │   ├── __init__.py
    │   ├── training_pipeline/
    │   │   └── __init__.py
    │   ├── application.py
    │   ├── database.py
    │   ├── env_variables.py
    │   └── s3_bucket.py
    ├── data_access/
    │   ├── __init__.py
    │   └── heart_stroke_data.py
    ├── entity/
    │   ├── __init__.py
    │   ├── artifact_entity.py
    │   ├── config_entity.py
    │   ├── estimator.py
    │   └── s3_estimator.py
    ├── exception/
    │   └── __init__.py
    ├── logger/
    │   └── __init__.py
    ├── pipeline/
    │   ├── __init__.py
    │   ├── train_pipeline.py
    │   └── prediction_pipline.py
    └── utils/
        ├── __init__.py
        └── main_utils.py
```

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections
![image](https://user-images.githubusercontent.com/57321948/193536736-5ccff349-d1fb-486e-b920-02ad7974d089.png)

## Project Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p stroke python=3.9 -y
```

```bash
conda activate stroke/
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the  environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://<username>:<password>@ineuron-ai-projects.7eh1w4s.mongodb.net/?retryWrites=true&w=majority"

```

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train

```

### Step 7. Prediction application
```bash
http://localhost:8080/predict

```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> --build-arg MONGODB_URL=<MONGODB_URL> . -t <tag>

```
3. Run the Docker image
```
docker run -d -p 8080:8080 <IMAGEID>
```
## Models Used 
- Logistic Regression
- KNeighbors Classifier
- XGB Classifier
- CatBoost Classifier
- SVC
- AdaBoost Classifier
- RandomForest Classifier

From these above models after hyperparameter optimization we selected Top two models which were KNeighbors Classifier and Random Forest Classifier and used the following in Pipeline.

GridSearchCV is used for Hyperparameter Optimization in the pipeline.

heart_stroke is the main package folder which contains all codes.


#### Conclusion 
- This Project can be used in real-life by Health Clinics to predict if the user has chance of heart stroke or not.
- Can be implemented in hospital website to predict the chance of heart stroke for the patients.
- As heart diseases and strokes are increasing rapidly across the world and causing deaths, it becomes necessary to develop an efficient system that would predict the heart stroke effectively before hand so that immediate medical attention can be given. In the proposed system, the most effective algorithm for stroke prediction was obtained after comparative analysis of the accuracy scores of various models.
