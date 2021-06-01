import pandas as pd
import sklearn
import os

print(os.environ.get("AIP_DATA_FORMAT"))
print(os.environ.get("AIP_TRAINING_DATA_URI"))
print(os.environ.get("AIP_VALIDATION_DATA_URI"))
print(os.environ.get("AIP_TEST_DATA_URI"))
