import datetime
import pathlib

import coolname


class Workspace:
    """
    Args:
        ...
    """

    def __init__(self, root: str, project: str, run_name: str = None, timestamp: bool = False):
        self.root = pathlib.Path(root)
        self.project = project
        self.run_name = run_name if run_name else coolname.generate_slug(2)
        if timestamp:
            time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.run_name = f"{time_stamp}-{self.run_name}"
        self.run_dir = self.root / project / self.run_name

    def __repr__(self):
        return f"Workspace(run_dir={self.run_dir})"

    @property
    def project_dir(self):
        return pathlib.Path(self.root / self.project)

    def mkdir(self, path=None):
        if path is None:
            pathlib.Path.mkdir(self.run_dir, parents=True, exist_ok=True)
        else:
            pathlib.Path.mkdir(path, parents=True, exist_ok=True)
