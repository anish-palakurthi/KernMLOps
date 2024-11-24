import subprocess
from dataclasses import dataclass
from typing import cast

from data_schema import GraphEngine, demote
from kernmlops_benchmark.benchmark import Benchmark, GenericBenchmarkConfig
from kernmlops_benchmark.errors import (
  BenchmarkNotInCollectionData,
  BenchmarkNotRunningError,
  BenchmarkRunningError,
)
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class MongoDbConfig(ConfigBase):
  record_count: int = 1000000
  readProportion: float = 0.25
  updateProportion: float = 0.75


class MongoDbBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "mongodb"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return MongoDbConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        mongodb_config = cast(MongoDbConfig, getattr(config, cls.name()))
        return MongoDbBenchmark(generic_config=generic_config, config=mongodb_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: MongoDbConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / self.name()
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return True
        # print(f'is_configured directory name: {self.benchmark_dir}')
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()

        bash_file_path = "scripts/run_benchmarks/run_mongodb.sh"
        self.process = subprocess.Popen(
            [
                "bash",
                str(bash_file_path),
                str(self.config.record_count),
                str(self.config.readProportion),
                str(self.config.updateProportion)
            ],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )

    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        return self.process.poll()

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.wait()

    def kill(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.terminate()

    @classmethod
    def plot_events(cls, graph_engine: GraphEngine) -> None:
        if graph_engine.collection_data.benchmark != cls.name():
            raise BenchmarkNotInCollectionData()
        # TODO(Patrick): plot when a trial starts/ends
