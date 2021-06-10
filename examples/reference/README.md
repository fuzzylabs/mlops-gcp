## Preparation
```
python -m venv env/
source env/bin/activate
pip install -r requirements.txt
dvc pull
```

## Step 0 -- locally train a model
## Step 1 -- locally train with DVC pipeline
* Copy `dvc-step1.yaml` to `dvc.yaml`
* `dvc repro` to run training

### Experiment tracking

This step uses [Sacred](https://github.com/IDSIA/sacred) for experiment tracking. The results are saved to MongoDB.
To run MongoDB locally with Docker:

```
docker run --name mongo -p 27017:27017 -d mongo:latest
```

To view the experiments, you can use [Omniboard](https://vivekratnavel.github.io/omniboard/#/README). To start Omniboard locally with the local MongoDB:
```
docker run -it --rm -p 9000:9000 --network host --name omniboard vivekratnavel/omniboard -m localhost:27017:sacred
```

and open dashboard in the browser [http://localhost:9000](http://localhost:9000)
## Step 2 -- train on Vertex AI
* Copy `dvc-step2.yaml` to `dvc.yaml`
* `dvc repro` to run training
## Step 3 -- train and deploy on Vertex AI
* Copy `dvc-step3.yaml` to `dvc.yaml`
* `dvc repro` to run training
* `cd step3 && python3 deploy-vertex.py` to deploy