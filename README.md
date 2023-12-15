# Kidney-Disease-Classification-DeepLearning-Project


## Workflows

1. Update config.yaml
2. Update secrets.yaml [optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py


# How to run?

### Steps:
Clone the repository

```bash
https://github.com/SouravHalder1996/Kidney-Disease-Classification-DeepLearning-Project.git
```

### Step-01: Create a conda environment after opening the repository

```bash
conda create -n kidney-disease python=3.8 -y
```

```bash
conda activate kidney-disease
```


### Step-02: Install the requirements

```bash
pip install -r requirements.txt
```


### Step-03: Export these below environment variables by running these commands inside the terminal

```bash
export MLFLOW_TRACKING_URI=<copied "MLFLOW_TRACKING_URI" from DagsHub> 
export MLFLOW_TRACKING_USERNAME=<copied "MLFLOW_TRACKING_USERNAME" from DagsHub> 
export MLFLOW_TRACKING_PASSWORD=<copied "MLFLOW_TRACKING_PASSWORD" from DagsHub> 
```



