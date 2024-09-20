import logging
import os
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import cast
from urllib.parse import urlparse

from pygit2 import clone_repository, discover_repository
from pygit2.repository import Repository
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

CACHE_DIR_BASE = (
    Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "deepspan_deinterleave_datasets_subjects"
)
SUBJECTS = [
    "https://github.com/apache/hadoop",
    "https://github.com/apache/hive",
    "https://github.com/apache/hbase",
    "https://github.com/apache/lucene",
    "https://github.com/apache/tomcat",
    "https://github.com/apache/activemq",
    "https://github.com/apache/pig",
    "https://github.com/apache/xmlgraphics-fop",
    "https://github.com/apache/logging-log4j2",
    "https://github.com/apache/ant",
    "https://github.com/apache/struts",
    "https://github.com/apache/jmeter",
    "https://github.com/apache/karaf",
    "https://github.com/apache/zookeeper",
    "https://github.com/apache/mahout",
    "https://github.com/apache/openmeetings",
    "https://github.com/apache/maven",
    "https://github.com/apache/pivot",
    "https://github.com/apache/empire-db",
    "https://github.com/apache/mina",
    "https://github.com/apache/creadur-rat",
]

logging.basicConfig(format=r"%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    with logging_redirect_tqdm():
        if os.getenv("DEBUG"):
            logger.setLevel(logging.DEBUG)
        for project in tqdm(subjects()):
            tqdm.write(str(project.path))


def traceit[**A, B](f: Callable[A, B]) -> Callable[A, B]:
    def wrapper(*args: A.args, **kwargs: A.kwargs) -> B:
        logger.debug("trace f=%s called with %s %s", f, args, kwargs)
        retval = f(*args, **kwargs)
        logger.debug("trace f=%s returned %s", f, retval)
        return retval

    return wrapper


clone_repository = traceit(clone_repository)
discover_repository = traceit(discover_repository)
rmtree = traceit(rmtree)


@traceit
def cache_dir(*name: str) -> Path:
    path = CACHE_DIR_BASE / Path(*name)
    path.mkdir(parents=True, exist_ok=True)
    return path


@traceit
def cache_repo(path: Path) -> Repository | None:
    try:
        if discover_repository(str(path)) is None:
            return None
        return Repository(str(path))
    except ValueError:
        return None


@dataclass
class Project:
    name: str
    path: Path
    repo: Repository

    @classmethod
    @traceit
    def clone(cls, url: str) -> "Project":
        parsed = urlparse(url)
        name = parsed.path.strip("/")
        path = cache_dir(name)
        if (repo := cache_repo(path)) is None:
            rmtree(path, ignore_errors=True)
            repo = cast(Repository, clone_repository(url, str(path), depth=0))
        return cls(name, path, repo)


class ProjectCloner:
    subjects: Sequence[str]

    def __init__(self, subjects: Sequence[str]):
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.subjects)

    @traceit
    def __iter__(self) -> Generator[Project, None, None]:
        with ThreadPoolExecutor() as executor:
            for future in as_completed(map(partial(executor.submit, Project.clone), self.subjects)):
                yield future.result()


@traceit
def subjects() -> ProjectCloner:
    """yields repositories subject in the study"""
    return ProjectCloner(SUBJECTS)


if __name__ == "__main__":
    main()
