from compiler_gym.datasets import Benchmark, Dataset, BenchmarkUri
from typing import Iterable

class CBenchDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://gcc_pr-cbench",
            license="N/A",
            description="CBench 1.1 dataset",
            validatable="Yes",
        )

    def benchmark_uris(self) -> Iterable[str]:
        return ["Whatever you want"]

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        return Benchmark.from_file_contents(uri, None)
