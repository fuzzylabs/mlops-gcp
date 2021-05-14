from typing import Optional
from project import ProjectDefinition


def init(name: Optional[str] = None, gcp_project: Optional[str] = None, gcp_region: Optional[str] = None,
         gcp_bucket_name: Optional[str] = None, machine_type: Optional[str] = None,
         executor_image: Optional[str] = None, python_module: Optional[str] = None):
    _name = ""
    if name is None:
        _name = input("Display name: ")
    else:
        _name = name

    _gcp_project = ""
    if gcp_project is None:
        _gcp_project = input("GCP project: ")
    else:
        _gcp_project = gcp_project

    _gcp_region = ""
    if gcp_region is None:
        _gcp_region = input("GCP region: ")
    else:
        _gcp_region = gcp_region

    _gcp_bucket_name = ""
    if gcp_bucket_name is None:
        _gcp_bucket_name = input("Google Storage bucket name: ")
    else:
        _gcp_bucket_name = gcp_bucket_name

    _machine_type = ""
    if machine_type is None:
        _machine_type = input("Machine type [n2-standard-4]: ")
        if _machine_type == "":
            _machine_type = "n2-standard-4"
    else:
        _machine_type = machine_type

    _executor_image = "europe-docker.pkg.dev/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest"
    print("Executor image:", _executor_image)

    _python_module = ""
    if python_module is None:
        _python_module = input("Python module: ")
    else:
        _python_module = python_module

    config = ProjectDefinition(
        name=_name,
        gcp_project=_gcp_project,
        gcp_region=_gcp_region,
        gcp_bucket_name=_gcp_bucket_name,
        machine_type=_machine_type,
        executor_image=_executor_image,
        python_module=_python_module,
    )
    print(config)
    config.to_file("./project.yaml")
