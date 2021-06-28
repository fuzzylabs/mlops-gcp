# from google.cloud.logging_v2 import Client
# from google.cloud.logging_v2 import Resource
from scipy.stats import kstest
from sklearn.decomposition import PCA

# logging_client = Client()
# resource = Resource(
#     type="aiplatform.googleapis.com/Endpoint",
#     labels={
#         "resource_container": "fuzzylabs",
#         "location": "europe-west4",
#         "endpoint_id": "6923967767733338112"
#     }
# )
# logger = logging_client.logger("aiplatform.googleapis.com/prediction_container")
# print("Between this")
# logger.log_struct({
#     "test": "test"
# }, resource=resource)
# print("And this")
# logging_handler = logging_client.get_default_handler()
# logging_client.setup_logging()
# logger = logging_client.logger("Monitoring")


class PCAMonitoring:
    def __init__(self, pca: PCA, train_data_pca):
        self.pca = pca
        self.train_data_pca = train_data_pca
        self.data = []
        self.monitoring_trigger_number = 100

    def get_distances(self, data):
        """
        Calculates Kolmogorov-Smirnov distance and p-values for each PC
        :param data:
        :return: Array of tuples (KS distance, p-value)
        """
        data_pca = self.pca.transform(data)
        return [list(kstest(self.train_data_pca[:, i], data_pca[:, i])) for i in range(self.pca.n_components)]

    def add_data(self, data):
        """

        :param data: 2D array of input data of shape (n_samples, dimensions)
        :return:
        """
        self.data += data
        if len(self.data) >= self.monitoring_trigger_number:
            distances = self.get_distances(self.data)
            drifted_pcs = [i for i in range(self.pca.n_components) if distances[i][1] < 0.05]
            level = "INFO" if len(drifted_pcs) == 0 else "WARNING"
            # logger.log_struct({
            #     "severity": level,
            #     "kstest": distances,
            #     "drifted_pcs": drifted_pcs,
            # }, severity=level)
            self.data = []
