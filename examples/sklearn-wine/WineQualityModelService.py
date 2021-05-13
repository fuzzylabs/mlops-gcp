import bentoml
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.adapters import JsonInput


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([SklearnModelArtifact('model')])
class WineQualityModelService(bentoml.BentoService):

    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, input: dict):
        """
        AI Platform Unified expects a Json Input with the following structure:
        {
          "instances": INSTANCES,
          "parameters": PARAMETERS
        }
        See more here: https://cloud.google.com/ai-platform-unified/docs/predictions/custom-container-requirements#request_requirements
        And returns a JSON Dict with the following structure:
        {'predictions': PREDICTIONS}
        See more here: https://cloud.google.com/ai-platform-unified/docs/predictions/custom-container-requirements#prediction
        """
        return {'predictions': self.artifacts.model.predict(input['instances'])}
