import logging
import os
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from pygit2 import clone_repository, discover_repository
from pygit2.enums import ResetMode
from pygit2.repository import Repository
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from shutil import rmtree

CACHE_DIR_BASE = (
    Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "deepspan_deinterleave_datasets_subjects"
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

logging.basicConfig(
    format=r"%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    with logging_redirect_tqdm():
        if os.getenv("DEBUG"):
            logger.setLevel(logging.DEBUG)
        for project in tqdm(subjects()):
            tqdm.write(str(project.path))


def traceit[**A, B](f: Callable[A, B]) -> Callable[A, B]:
    def wrapper(*args: A.args, **kwargs: A.kwargs) -> B:
        logger.debug(f"trace {f=} called with {args=} {kwargs=}")
        retval = f(*args, **kwargs)
        logger.debug(f"trace {f=} returned {retval}")
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
def cache_repo(url, path: Path) -> Repository:
    try:
        if discover_repository(str(path)) is None:
            raise ValueError()
        repo = Repository(str(path))
        return repo
    except ValueError:
        rmtree(path, ignore_errors=True)
        repo = cast(Repository, clone_repository(url, str(path), depth=0))
        return repo


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
        repo = cache_repo(url, path)
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
            futures: list[Future[Project]] = []
            for subject in self.subjects:
                futures.append(executor.submit(Project.clone, subject))
            for future in as_completed(futures):
                yield future.result()


@traceit
def subjects() -> ProjectCloner:
    """yields repositories subject in the study"""
    return ProjectCloner(SUBJECTS)


if __name__ == "__main__":
    main()
