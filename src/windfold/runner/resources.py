"""what this machine offers the runner, and how the runner splits it into pools.

cpu jobs get whole physical cores, pinned to the first hardware thread of each core. gpu
workers use the cores' second hardware threads, minus the reserved ones, which stay free.
with cpu_jobs_on_second_threads, those second threads become more cpu slots instead, so two
jobs share each core: more done per hour, each job slower. a gpu worker spends its time in one
python thread feeding the device, so a busy sibling thread slows it; with
gpu_workers_keep_whole_cores, each gpu worker keeps a whole core. a cgroup cpu quota, as in a
container, caps the cores used to the cpus it allows; a cgroup memory limit caps the total
memory, and the limit less what its cgroup uses caps the available memory. memory is in gb of
2**30 bytes. torch is never imported here; a short probe process checks that the installed build
can use the detected devices.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from windfold.config import Resources

CPU_DIR = Path("/sys/devices/system/cpu")
MEMINFO = Path("/proc/meminfo")
CGROUP_ROOT = Path("/sys/fs/cgroup")
CGROUP_SELF = Path("/proc/self/cgroup")
GPU_FIELDS = "index,uuid,name,memory.total,compute_cap"
DEVICE_FIELDS = "index,memory.used,memory.total,utilization.gpu"
TORCH_PROBE = (
    "import json, torch\n"
    "print(json.dumps({'available': torch.cuda.is_available(), "
    "'count': torch.cuda.device_count(), 'arch': torch.cuda.get_arch_list()}))\n"
)


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_gb: float
    compute_capability: str


@dataclass(frozen=True)
class Hardware:
    cores: tuple[tuple[int, ...], ...]
    memory_gb: float
    available_memory_gb: float
    gpus: tuple[Gpu, ...]
    cpu_quota: float | None = None


@dataclass(frozen=True)
class Allocation:
    cpu_slots: tuple[tuple[int, ...], ...]
    gpu_threads: tuple[int, ...]
    reserved_threads: tuple[int, ...]
    gpus: tuple[Gpu, ...]
    gpu_workers_per_device: int
    automl_jobs_per_device: int
    memory_budget_gb: float
    notes: tuple[str, ...]
    shared_cores: bool = False


def cpu_list(text: str) -> list[int]:
    numbers: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        numbers.extend(range(int(first), int(last or first) + 1))
    return numbers


def physical_cores(threads: Iterable[int], cpu_dir: Path = CPU_DIR) -> tuple[tuple[int, ...], ...]:
    """usable threads grouped by physical core: each core's threads ascending, cores in order."""
    usable = set(threads)
    cores: set[tuple[int, ...]] = set()
    for thread in usable:
        topology = cpu_dir / f"cpu{thread}" / "topology"
        siblings_file = topology / "core_cpus_list"
        if not siblings_file.exists():
            siblings_file = topology / "thread_siblings_list"
        siblings = cpu_list(siblings_file.read_text()) if siblings_file.exists() else [thread]
        cores.add(tuple(sorted(t for t in siblings if t in usable)))
    return tuple(sorted(cores))


def quota_in(cpu_max: Path) -> float | None:
    try:
        quota, period = cpu_max.read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else int(quota) / int(period)


def cfs_quota_in(folder: Path) -> float | None:
    try:
        quota = int((folder / "cpu.cfs_quota_us").read_text())
        period = int((folder / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return quota / period if quota > 0 and period > 0 else None


def lines_of(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def cgroup_folders(root: Path, membership: Path) -> list[Path]:
    """the cgroup v2 folders over this process, from its own cgroup up to the root."""
    folders = []
    for line in lines_of(membership):
        _, controllers, path = line.split(":", 2)
        if controllers:
            continue
        folder = root / path.strip("/")
        while True:
            folders.append(folder)
            if folder == root or root not in folder.parents:
                break
            folder = folder.parent
    return folders


def cgroup_v1_folders(controller: str, membership: Path) -> list[Path]:
    """the cgroup v1 folders of controller over this process, from its own cgroup up to the top
    of the mounted hierarchy. the mount table beside membership says where the hierarchy is
    mounted and which of its cgroups the mount shows as that top, as a container's mount does."""
    mounts = []
    for line in lines_of(membership.with_name("mountinfo")):
        fields = line.split()
        tail = fields[fields.index("-") + 1 :] if "-" in fields else []
        if len(tail) == 3 and tail[0] == "cgroup" and controller in tail[2].split(","):
            mounts.append((fields[3], Path(fields[4])))
    paths = [
        parts[2]
        for parts in (line.split(":", 2) for line in lines_of(membership))
        if len(parts) == 3 and controller in parts[1].split(",")
    ]
    if not mounts or not paths:
        return []
    top, point = mounts[0]
    own = PurePosixPath(paths[0])
    folder = point / own.relative_to(top) if own.is_relative_to(top) else point
    return [folder, *(above for above in folder.parents if point in (above, *above.parents))]


def cpu_quota(root: Path = CGROUP_ROOT, membership: Path = CGROUP_SELF) -> float | None:
    """the cpus a cgroup quota lets this process use, or None when no quota limits it. quotas are
    read at every level from the process's own cgroup up, to the root under cgroup v2 and to the
    top of the mounted cpu hierarchy under cgroup v1, and the smallest wins."""
    found = [quota_in(folder / "cpu.max") for folder in cgroup_folders(root, membership)]
    found += [cfs_quota_in(folder) for folder in cgroup_v1_folders("cpu", membership)]
    limits = [limit for limit in found if limit is not None]
    return min(limits) if limits else None


def bytes_in(path: Path) -> int | None:
    """the byte count a cgroup file holds, or None when it is missing or says max."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def stat_bytes(memory_stat: Path, key: str) -> int:
    """one count of a cgroup's memory.stat, or 0 when it has none."""
    try:
        lines = memory_stat.read_text().splitlines()
    except OSError:
        return 0
    for line in lines:
        name, _, value = line.partition(" ")
        if name == key:
            return int(value)
    return 0


def memory_limits(root: Path, membership: Path) -> list[tuple[int, int]]:
    """each cgroup memory limit over this process and what its cgroup uses, in bytes; inactive
    file cache, which the kernel reclaims first, is not counted as used. limits are read at every
    level from the process's own cgroup up, to the root under cgroup v2 and to the top of the
    mounted memory hierarchy under cgroup v1."""
    places = [
        (folder, "memory.max", "memory.current", "inactive_file")
        for folder in cgroup_folders(root, membership)
    ]
    places += [
        (folder, "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")
        for folder in cgroup_v1_folders("memory", membership)
    ]
    limits = []
    for folder, limit_file, usage_file, inactive in places:
        limit = bytes_in(folder / limit_file)
        if limit is not None:
            usage = bytes_in(folder / usage_file) or 0
            limits.append((limit, max(usage - stat_bytes(folder / "memory.stat", inactive), 0)))
    return limits


def memory(
    meminfo: Path = MEMINFO, root: Path = CGROUP_ROOT, membership: Path = CGROUP_SELF
) -> tuple[float, float]:
    """total and available memory in gb: the host's, with each cgroup memory limit over this
    process capping the total, and the limit less what its cgroup uses capping what is free."""
    values = {}
    for line in meminfo.read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024 / 2**30
    total, available = values["MemTotal"], values["MemAvailable"]
    for limit, used in memory_limits(root, membership):
        total = min(total, limit / 2**30)
        available = min(available, max(limit - used, 0) / 2**30)
    return total, available


def nvidia_smi(fields: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [[cell.strip() for cell in line.split(",")] for line in result.stdout.splitlines()]


def visible(gpus: list[Gpu], setting: str | None) -> tuple[Gpu, ...]:
    """the devices CUDA_VISIBLE_DEVICES leaves visible, in its order; parsing stops at the first
    entry that names no device, as CUDA does."""
    if setting is None:
        return tuple(gpus)
    chosen: list[Gpu] = []
    for entry in (e.strip() for e in setting.split(",")):
        match = [
            g
            for g in gpus
            if (entry.isdigit() and g.index == int(entry))
            or (entry.startswith("GPU-") and g.uuid.startswith(entry))
        ]
        if len(match) != 1:
            break
        chosen.append(match[0])
    return tuple(chosen)


def detect_gpus() -> tuple[Gpu, ...]:
    gpus = [
        Gpu(int(index), uuid, name, float(total) / 1024, capability)
        for index, uuid, name, total, capability in nvidia_smi(GPU_FIELDS)
    ]
    return visible(gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))


def detect() -> Hardware:
    total, available = memory()
    return Hardware(
        cores=physical_cores(os.sched_getaffinity(0)),
        memory_gb=total,
        available_memory_gb=available,
        gpus=detect_gpus(),
        cpu_quota=cpu_quota(),
    )


def torch_problem(gpus: tuple[Gpu, ...]) -> str | None:
    """why the installed torch build cannot run on these devices, or None when it can."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(g.uuid for g in gpus)}
    try:
        probe = subprocess.run(
            [sys.executable, "-c", TORCH_PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "the torch probe did not finish in 300 s"
    if probe.returncode != 0:
        lines = probe.stderr.strip().splitlines()
        return f"torch could not start: {lines[-1] if lines else probe.returncode}"
    report = json.loads(probe.stdout.strip().splitlines()[-1])
    if not report["available"] or report["count"] != len(gpus):
        return f"torch sees {report['count']} of {len(gpus)} devices"
    arch = report["arch"]
    for gpu in gpus:
        major, minor = gpu.compute_capability.split(".")
        native = f"sm_{major}{minor}" in arch
        jit = any(
            a.startswith("compute_") and int(a.removeprefix("compute_")) <= int(major + minor)
            for a in arch
        )
        if not (native or jit):
            return f"torch is built for {arch}, not compute capability {gpu.compute_capability}"
    return None


def allocate(
    hardware: Hardware,
    resources: Resources,
    *,
    cpu_jobs: int | None,
    reserved_memory_gb: float,
    gpus: tuple[Gpu, ...],
) -> Allocation:
    """split the machine: cpu slots of whole cores on first threads, gpu workers on the second
    threads, the last second threads reserved. cpu_jobs caps the slots; None takes all that fit.
    a cpu quota keeps only as many cores as it has whole cpus, and at least one. with
    cpu_jobs_on_second_threads, slots on the unreserved second threads follow the first-thread
    slots once every gpu worker keeps a second thread, and gpu workers keep the rest. with
    gpu_workers_keep_whole_cores, the gpu workers keep the last cores, one each, both threads
    but the reserved ones, and the cpu slots come from the cores before them."""
    cores = hardware.cores
    if not cores:
        raise ValueError("no usable cpu cores")
    if cpu_jobs is not None and cpu_jobs < 1:
        raise ValueError(f"cpu_jobs must be at least 1, not {cpu_jobs}")
    notes: list[str] = []
    if hardware.cpu_quota is not None and int(hardware.cpu_quota) < len(cores):
        kept = max(int(hardware.cpu_quota), 1)
        notes.append(
            f"a cgroup cpu quota of {hardware.cpu_quota:g} cpus leaves {kept} of {len(cores)} cores"
        )
        cores = cores[:kept]
    width = min(resources.cores_per_cpu_job, len(cores))
    if width < resources.cores_per_cpu_job:
        notes.append(f"only {len(cores)} cores usable, so cpu jobs get {width} cores each")
    seconds = [thread for core in cores for thread in core[1:]]
    keep = max(len(seconds) - resources.reserved_threads, 0)
    reserved = tuple(seconds[keep:])
    gpu_threads = tuple(seconds[:keep])
    workers = resources.gpu_workers_per_device * len(gpus)
    whole = resources.gpu_workers_keep_whole_cores and workers > 0
    if whole and len(cores) - workers < width:
        notes.append(f"{len(cores)} cores are too few for {workers} gpu workers to keep whole ones")
        whole = False
    if whole:
        owned = cores[-workers:]
        cores = cores[:-workers]
        gpu_threads = tuple(sorted(t for core in owned for t in core if t not in reserved))
        pool = tuple(t for core in cores for t in core[1:] if t not in reserved)
        spare_seconds = len(pool)
        notes.append(f"each of {workers} gpu workers keeps a whole core")
    else:
        pool = gpu_threads
        spare_seconds = max(len(gpu_threads) - workers, 0)
    count = len(cores) // width
    shared = spare_seconds // width if resources.cpu_jobs_on_second_threads else 0
    if cpu_jobs is not None and cpu_jobs > count + shared:
        notes.append(f"{cpu_jobs} cpu jobs asked for, {count + shared} fit")
    if cpu_jobs is not None:
        count, shared = min(count, cpu_jobs), min(shared, max(cpu_jobs - count, 0))
    slots = tuple(
        tuple(core[0] for core in cores[k * width : (k + 1) * width]) for k in range(count)
    )
    slots += tuple(pool[k * width : (k + 1) * width] for k in range(shared))
    if not whole:
        gpu_threads = gpu_threads[shared * width :]
    if shared:
        notes.append(f"{shared} cpu slots on second threads, so their jobs share cores")
    if not gpu_threads:
        spare = [core[0] for core in cores[count * width :]]
        gpu_threads = tuple(spare or [core[0] for core in cores])
        notes.append("no second hardware threads free, so gpu workers share first threads")
    return Allocation(
        cpu_slots=slots,
        gpu_threads=gpu_threads,
        reserved_threads=reserved,
        gpus=gpus,
        gpu_workers_per_device=resources.gpu_workers_per_device,
        automl_jobs_per_device=resources.automl_jobs_per_device,
        memory_budget_gb=max(hardware.available_memory_gb - reserved_memory_gb, 0.0),
        notes=tuple(notes),
        shared_cores=shared > 0,
    )


def device_use() -> dict[int, tuple[float, float, float]]:
    """memory used and total in gb and percent busy, by device index."""
    return {
        int(index): (float(used) / 1024, float(total) / 1024, float(busy))
        for index, used, total, busy in nvidia_smi(DEVICE_FIELDS)
    }


def write_allocation(
    path: Path,
    hardware: Hardware,
    allocation: Allocation,
    profiles: dict[str, dict[str, float]],
    gpu_problem: str | None,
) -> None:
    record = {
        "hardware": asdict(hardware),
        "pools": asdict(allocation),
        "gpu_problem": gpu_problem,
        "profiles": profiles,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(partial, path)
