import time
from typing import List, Tuple, Optional
from threading import Thread, Event, Lock


class GPUMemProf:
    """
    GPU memory profiler that samples memory usage at regular intervals in a background thread.
    Works in both synchronous and asynchronous contexts.
    """

    def __init__(self, interval: float = 1.0):
        """
        Initialize the profiler with a sampling interval in seconds.

        Args:
            interval: Time between samples in seconds (default: 1.0)
        """
        self.interval = interval
        self._running_event: Event = Event()
        self._thread: Optional[Thread] = None
        self._lock: Lock = Lock()
        self.memory_samples: List[Tuple[float, List[Tuple[float, float]]]] = []

        # NVML state
        self._nvml = None
        self._device_handles: List[object] = []
        try:
            import pynvml as nvml  # type: ignore
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            self._device_handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            self._nvml = nvml
        except Exception:
            # No NVML/GPU available; sampling will yield empty entries
            self._nvml = None
            self._device_handles = []

    def get_vmem_usage(self) -> Optional[List[Tuple[float, float]]]:
        """
        Return per-GPU (used_mb, total_mb) using cached NVML handles.
        Returns None if NVML is unavailable.
        """
        if self._nvml is None or not self._device_handles:
            return None
        output: List[Tuple[float, float]] = []
        for handle in self._device_handles:
            info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            output.append((info.used / 1024**3, info.total / 1024**3))
        return output

    def _run_sampler(self) -> None:
        while self._running_event.is_set():
            try:
                timestamp = time.time()
                memory_usage = self.get_vmem_usage()
                sample: Tuple[float, List[Tuple[float, float]]] = (
                    timestamp,
                    memory_usage if memory_usage is not None else [],
                )
                with self._lock:
                    self.memory_samples.append(sample)
            except Exception as e:
                with self._lock:
                    self.memory_samples.append((time.time(), []))
            # Sleep but allow prompt stop
            self._running_event.wait(self.interval)

    def start(self) -> None:
        """
        Start sampling in a background thread.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self.memory_samples = []
        self._running_event.set()
        self._thread = Thread(target=self._run_sampler, daemon=True)
        self._thread.start()

    def stop(self) -> List[Tuple[float, List[Tuple[float, float]]]]:
        """
        Stop sampling and return the collected memory usage data.

        Returns:
            List of tuples containing (timestamp, memory_usage) for each sample
        """
        if self._thread is None:
            return []
        self._running_event.clear()
        self._thread.join()
        self._thread = None
        with self._lock:
            return list(self.memory_samples)

    def get_current_samples(self) -> List[Tuple[float, List[Tuple[float, float]]]]:
        """
        Get the current memory samples without stopping the profiler.

        Returns:
            List of tuples containing (timestamp, memory_usage) for each sample so far
        """
        with self._lock:
            return list(self.memory_samples)

    def is_running(self) -> bool:
        """
        Check if the profiler is currently running.

        Returns:
            True if profiling is active, False otherwise
        """
        return self._thread is not None and self._thread.is_alive()


class SystemMemProf:
    """
    System memory profiler that samples current process RSS (in MB) at regular intervals
    in a background thread. Optionally includes child process memory.
    """

    def __init__(self, interval: float = 1.0, include_children: bool = True, pid: Optional[int] = None):
        """
        Initialize the profiler.

        Args:
            interval: Sampling interval in seconds
            include_children: If True, include RSS of child processes
            pid: Target process id. Defaults to current process if None
        """
        self.interval = interval
        self.include_children = include_children
        self._running_event: Event = Event()
        self._thread: Optional[Thread] = None
        self._lock: Lock = Lock()
        self.memory_samples: List[Tuple[float, float]] = []

        # psutil state
        self._psutil = None
        self._process = None
        try:
            import psutil as _ps
            self._psutil = _ps
            self._process = _ps.Process(pid) if pid is not None else _ps.Process()
        except Exception:
            self._psutil = None
            self._process = None

    def _get_rss_mb(self) -> Optional[float]:
        if self._psutil is None or self._process is None:
            return None
        try:
            rss = float(self._process.memory_info().rss)
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        rss += float(child.memory_info().rss)
                    except Exception:
                        continue
            return rss / (1024.0 ** 3)
        except Exception:
            return None

    def _run_sampler(self) -> None:
        while self._running_event.is_set():
            timestamp = time.time()
            value_mb = self._get_rss_mb()
            with self._lock:
                self.memory_samples.append((timestamp, value_mb if value_mb is not None else 0.0))
            self._running_event.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self.memory_samples = []
        self._running_event.set()
        self._thread = Thread(target=self._run_sampler, daemon=True)
        self._thread.start()

    def stop(self) -> List[Tuple[float, float]]:
        if self._thread is None:
            return []
        self._running_event.clear()
        self._thread.join()
        self._thread = None
        with self._lock:
            return list(self.memory_samples)

    def get_current_samples(self) -> List[Tuple[float, float]]:
        with self._lock:
            return list(self.memory_samples)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()