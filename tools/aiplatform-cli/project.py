from dataclasses import dataclass
from yaml import load, dump, SafeLoader, YAMLObject


@dataclass
class ProjectDefinition(YAMLObject):
    yaml_tag = "!ProjectDefinition"
    yaml_loader = SafeLoader

    name: str
    gcp_project: str
    gcp_region: str
    gcp_bucket_name: str
    machine_type: str
    executor_image: str
    python_module: str

    def to_file(self, filename):
        lines = dump(self)
        with open(filename, "w") as f:
            for line in lines:
                f.write(line)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            lines = list(f)
            yaml = "".join(lines)
            loaded = load(yaml, Loader=SafeLoader)
            return loaded


