
# Sea creature classifier

## Problem statement

This is the capstone project for DataTalksClub's Machine Learning Zoomcamp. I am using the [Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) on Kaggle. My kid loves sea creatures and the ocean, and I was also inspired by the [Fathomnet project](https://fathomnet.org).

The Fathomnet project is particularly interesting, because it "[enables] the future of conservation, exploration, and discovery" of the ocean (from their website). The ocean is one of the final frontiers for discovery, and the more we know about it, the more we can understand how to take care of it better. For example, a lot of research is going into determining the levels of ocean pollution and its impact on all life. However, it's not so easy for people to navigate the deep oceans, so organizations such as [MBARI](https://www.mbari.org/) build underwater robots to capture images and video, as well as map the ocean floor. AI can help us understand what's in the captured data.

In this project, I am building a simple sea creature classifier. Given an image, my web service will determine what kind of sea creature it is and try to classify it as one of the 23 creature classes in the dataset.


## Data

### Dataset
This project uses the [Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)

* 23 different sea animal classes
* At most 300px in height or width
* JPEG files


### Exploratory Data Analysis
In the exploratory data analysis, I did a few things:

* Remove duplicate images, to reduce overfitting or bias
* Add padding to images to prevent distortion when resizing as input to the model
* Remove some categories - mainly animals that don't live entirely underwater, as well as the Puffers category (puffers are a type of a fish and there's a category for fish already)

### TODO

rough notes
* EDA: analyze content of images, distribution of classes, distribution of images per class, etc.
  * remove duplicate images
* train model
  * split into train/test
  * variations
    * small (150x150) and large (299x299)
    * dropout vs no dropout
    * extra inner layers vs not
  * tuning
    * learning rate
    * dropout rate
    * augmentation
    * size of inner layers
  * make sure to include graphs!
* use SaturnCloud to train actual model
* convert model to tflite
* export training code to train.py
  * train.py should train only the best model
 

## Important files in this project

 

# How to run the model API

The commands provided in this README have been tested in Mac OS, and should also work in Linux and WSL.

## Install dependencies

Prerequisites: Python 3.11+, `pipenv`

Install pip dependencies
```
pipenv install
```

If you want to run the notebook, you'll also want to install the dev dependencies
```
pipenv install --dev
```

## Jupyter notebook

The notebook contains the data cleaning, analysis, and model training.

To run the notebook:
```
pipenv run jupyter notebook
```

## Run server locally

The trained model can be deployed in a web service (a Flask API). To start the server:

```
pipenv run gunicorn --bind 0.0.0.0:8080 predict:app
```

## Sending test data to server




### Sample data

```json

```

### Using postman

You can use [Postman](https://www.postman.com/downloads/) to send requests to the API.

![postman](img/postman_local.png)

### Using curl

If you prefer to issue requests via commandline, you can use `curl`.

```
TBD
```

## Docker

Docker provides a consistent way to package and deploy the API and model.

Build the Docker image:
```
docker build -t sea-class .
```

Run the Docker container:
```
docker run -it -p 8080:8080 sea-class:latest
```

You can issue requests directly to server running in the Docker container. See [Sending test data to server](#sending-test-data-to-server) section for an example request.

## Cloud deployment

I've deployed the model to AWS Elastic Beanstalk.

### Deployment details

```
export AWS_PROFILE=<aws-profile-name>
export AWS_EB_PROFILE=<aws-profile-name>

eb init
eb create sea-class

eb deploy
```

You can send prediction requests (`POST`) to TBD ENDPOINT.

### Using postman
![postman](img/postman_cloud.png)

### Using curl
```
TBD
```

Response:
```json
TBD
```