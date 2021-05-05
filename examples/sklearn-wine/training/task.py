from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_wine
import joblib
import pandas as pd
import numpy as np
import argparse
import os


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir',
                        dest='model_dir', default=os.getenv("AIP_MODEL_DIR"))

    args = parser.parse_args()
    print(args.model_dir)

    data, y = load_wine(as_frame=True, return_X_y=True)
    data["quality"] = y
    print(data)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    # %%

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = 0.5
    l1_ratio = 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Dump model
    if args.model_dir.startswith("gs://"):
        with open('model.joblib', 'wb') as f:
            joblib.dump(lr, f)
        from google.cloud import storage
        from google.cloud.storage.blob import Blob
        client = storage.Client()
        print(client)
        blob = Blob.from_string(os.path.join(args.model_dir, 'model.joblib'), client)
        print(blob)
        blob.upload_from_filename("model.joblib")
    else:
        with open(os.path.join(args.model_dir, 'model.joblib'), 'wb') as f:
            joblib.dump(lr, f)
