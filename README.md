# NLP - News classification

Train and deploy a news classifier based on ULMFit.

- Live version: https://nlp.imadelhanafi.com
- Serverless version: https://newsclassifier-eebuzelyaa-uc.a.run.app/
- Blog post: https://imadelhanafi.com/posts/text_classification_ulmfit/

<a href="https://nlp.imadelhanafi.com">
<img src="https://imadelhanafi.com/data/draft/nlp.png" width="500" height="400" class="center"/>
</a>

# Running on cloud/local machine

To run the application, we can use the pre-build docker image available on Docker Hub and simply run the following command

```bash
docker run --rm -p 8080:8080 imadelh/news:v1
```

The application will be available on http://0.0.0.0:8080.
The user can run a customized Gunicorn command to specify number of workers or an HTTPS certificate.

```bash
# Get into the container
docker run -it --rm -v ~/nlp:/cert -p 8080:8080 imadelh/news:v1 bash

# Run Gunicorn with specefic number of workers/threads
gunicorn --certfile '/path_to/chain.pem' --keyfile '/path_to/key.pem' --workers=4 --bind 0.0.0.0:8080 wsgi:app
```

# Serverless deployement - Google Run 

Google Run is a new service from GCP that allows serverless deployment of containers with HTTPS endpoints. The app will run on 1 CPU with 2GB memory and have the ability to scale automatically depending on the number of concurrent requests. 

- Build image and push it to Container Registry 

From a GCP project, we will use Google Shell to build the image and push it to GCR (container registry).

```
# Get name of project 
# For illustration we will call it PROJECT-ID

gcloud config get-value project
```

Create the following Dockerfile in your CloudShell session.

```
FROM imadelh/news:v_1cpu

# Google Run uses env variable PORT 

CMD gunicorn --bind :$PORT wsgi:app
```

Finally, we can build and submit the image to GCR.

```
gcloud builds submit --tag gcr.io/PROJECT-ID/news_classifier
```

- Deploy on Google Run


From Google Run page, we will use the image `gcr.io/PROJECT-ID/news_classifier:latest` to run the app. Create a new service 

<img src="https://imadelhanafi.com/data/draft/run.png" width="50%" height="50%">

Then enter the address of the image, choose other parameters as follows and deploy 

<img src="https://imadelhanafi.com/data/draft/run1.png" width="60%" height="60%">

After few seconds,  you will see a link to the app. 

<img src="https://imadelhanafi.com/data/draft/run3.png" width="50%" height="50%">

Serverless version may suffer from [**cold-start**](https://github.com/ahmetb/cloud-run-faq#cold-starts) if the service does not receive requests for a long time. 


# Reproduce results

## LR and SVM

- Requirements

To reproduce results reported in the blog post, we need to install the requirements in our development environment.

```bash
# Open requirement.txt and select torch==1.1.0 instead of the cpu version used for inference only.
# Then install requirements
pip install -r requirements.txt
```

- Hyper-parameter search

After completing the installation, we can run parameters search or training of sklearn models as follows

```bash
# Params search for SVM
cd sklearn_models
python3 params_search.py --model svc --exp_name svmsearch_all --data dataset_processed

# Params search for LR
python3 params_search.py --model lreg --exp_name logreg_all --data dataset_processed
```

The parameters space is defined in the file `sklearn_models/params_search.py`. The outputs will be saved in the logs folder.

- Training

Training a model for a fixed set of parameters can be done using `sklearn_models/baseline.py`

```bash
# Specify the parameters of the model inside baseline.py and run
python3 baseline.py --model svc --exp_name svc_all --data dataset_processed
```

The logs/metrics on test dataset will be saved in `sklearn_models/logs/` and the trained model will be saved in `sklearn_models/saved_models/`.


## ULMFit

To reproduce/train ULMFit model, the notebooks available in `ulmfit/` are used. Same requirements are needed as explained before. We will need a GPU to fine-tune LM models, this can be done using Google Colab.

- Notebook contents:

  - data preparation
  - Fine-tune ULMFit
  - Train ULMFit classifier
  - Predictions and evaluation
  - Exporting the trained model
  - Inference on CPU  


To be able to run the training, we need to specify the path to a folder where the training data is stored.

- Locally:

Save data from `data/`, then specify the absolute PATH in the beginning of the notebook.
```bash
# This is the absolute path to where folder "data" is available
PATH = "/app/analyse/"
```

- Google Colab:

Save the data in Google drive folder, for example `files/nlp/`

```bash
# The folder 'data' is saved in Google drive in "files/nlp/"
# While running the notebook from google colab, mount the drive and define PATH to data
from google.colab import drive
drive.mount('/content/gdrive/')

# then give the path where your data is stored (in google drive)
PATH = "/content/gdrive/My Drive/files/nlp/"
```

`01_ulmfit_balanced_dataset.ipynb` <a href="https://colab.research.google.com/github/imadelh/NLP-news-classification/blob/master/ulmfit_model/01_ulmfit_balanced_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> - Train ULMfit on balanced dataset


`02_ulmfit_all_data.ipynb` <a href="https://colab.research.google.com/github/imadelh/NLP-news-classification/blob/master/ulmfit_model/02_ulmfit_all_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> - Train ULMFit on full dataset


## Performance

Performance of ULMFit on the test dataset `data/dataset_inference` (see end of `02_ulmfit_all_data.ipynb` for the definition of test dataset).

```bash
# ULMFit - Performance on test dataset
            precision    recall  f1-score   support
micro avg                           0.73     20086
macro avg       0.66      0.61      0.63     20086
weighted avg    0.72      0.73      0.72     20086

Top 3 accuracy on test dataset:
0.9044
```

Trained model is available for download at: https://github.com/imadelh/NLP-news-classification/releases/download/v1.0/ulmfit_model

This project is a very basic text classifier. Here is a list of other features that could be added
- Feedback option to allow the user to submit a correction of the prediction.
- Fine-tune the model periodically based on new feedbacks.
- Compare performance to other language models (BERT, XLNet, etc).

---

Imad El Hanafi
