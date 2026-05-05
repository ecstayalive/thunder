from __future__ import annotations

import datetime
import pathlib

import coolname


class Workspace:
    """
    Args:
        ...
    """

    def __init__(
        self, root: str, project: str, run_name: str = None, timestamp: bool = False
    ):
        self.root = pathlib.Path(root)
        self.project = project
        self.run_name = run_name if run_name else coolname.generate_slug(2)
        if timestamp:
            time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.run_name = f"{time_stamp}-{self.run_name}"

    def __repr__(self):
        return f"Workspace(run_dir={self.run_dir})"

    @property
    def project_dir(self):
        return pathlib.Path(self.root / self.project)

    @property
    def run_dir(self):
        return self.root / self.project / self.run_name

    @property
    def checkpoint_dir(self):
        return self.run_dir / "checkpoints"

    def mkdir(self, path=None):
        if path is None:
            pathlib.Path.mkdir(self.run_dir, parents=True, exist_ok=True)
        else:
            pathlib.Path.mkdir(path, parents=True, exist_ok=True)

    def rm(self, path=None):
        if path is None:
            if self.run_dir.exists():
                for item in self.run_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        self.rm(item)
                self.run_dir.rmdir()
        else:
            if path.exists():
                for item in path.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        self.rm(item)
                path.rmdir()

    @classmethod
    def factory(cls, path: str | pathlib.Path) -> Workspace: ...
