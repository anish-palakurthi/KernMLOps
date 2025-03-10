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
class MemcachedConfig(ConfigBase):
  operation_count: int = 1000000
  read_proportion: float = 0.99
  update_proportion: float = 0.01
  insert_proportion: float = 0.00
  rmw_proportion: float = 0.00
  scan_proportion: float = 0.00
  delete_proportion: float = 0.00


class MemcachedBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "memcached"

    @classmethod
    def default_config(cls) -> ConfigBase:
        return MemcachedConfig()

    @classmethod
    def from_config(cls, config: ConfigBase) -> "Benchmark":
        generic_config = cast(GenericBenchmarkConfig, getattr(config, "generic"))
        memcached_config = cast(MemcachedConfig, getattr(config, cls.name()))
        return MemcachedBenchmark(generic_config=generic_config, config=memcached_config)

    def __init__(self, *, generic_config: GenericBenchmarkConfig, config: MemcachedConfig):
        self.generic_config = generic_config
        self.config = config
        self.benchmark_dir = self.generic_config.get_benchmark_dir() / "ycsb"
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        self.generic_config.generic_setup()

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()

        self.process = subprocess.Popen(
            [
                f"{self.benchmark_dir}/YCSB/bin/ycsb",
                "run",
                "memcached",
                "-s",
                "-P",
                f"{self.benchmark_dir}/YCSB/workloads/workloada",
                "-p",
                f"operationcount={self.config.operation_count}",
                "-p",
                "memcached.hosts=localhost:11211",
                "-p",
                f"readproportion={self.config.read_proportion}",
                "-p",
                f"updateproportion={self.config.update_proportion}",
                "-p",
                f"insertproportion={self.config.insert_proportion}",
                "-p",
                f"readmodifywriteproportion={self.config.rmw_proportion}",
                "-p",
                f"scanproportion={self.config.scan_proportion}",
                "-p",
                f"deleteproportion={self.config.delete_proportion}",
                "-p",
                "recordcount=1000000",
                "-p",
                "requestdistribution=zipfian"

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
