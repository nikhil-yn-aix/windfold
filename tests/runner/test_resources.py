from pathlib import Path

import pytest

from windfold.config import Resources
from windfold.runner.resources import (
    Gpu,
    Hardware,
    allocate,
    cpu_quota,
    memory,
    physical_cores,
    visible,
)

RESOURCES = Resources(
    data_dir="data",
    runs_dir="runs",
    reserved_threads=2,
    cores_per_cpu_job=2,
    gpu_workers_per_device=2,
    automl_jobs_per_device=2,
    status_interval_seconds=300,
    stall_minutes=20,
    replan_seconds=120,
    failure_fraction=0.02,
    failures_before_pause=3,
    memory_margin=1.25,
    reserved_memory_gb=8,
    load_rows=1_000_000,
    load_features=12,
)


def topology(tmp_path: Path, cores: int) -> Path:
    for thread in range(2 * cores):
        folder = tmp_path / f"cpu{thread}" / "topology"
        folder.mkdir(parents=True)
        core = thread % cores
        (folder / "core_cpus_list").write_text(f"{core},{core + cores}\n")
    return tmp_path


def split(cores, cpu_jobs=None, **changes):
    hardware = Hardware(cores=cores, memory_gb=64.0, available_memory_gb=60.0, gpus=())
    resources = RESOURCES.model_copy(update=changes)
    return allocate(hardware, resources, cpu_jobs=cpu_jobs, reserved_memory_gb=8.0, gpus=())


def test_cpu_jobs_get_whole_cores_and_gpu_workers_the_second_threads(tmp_path):
    cpu_dir = topology(tmp_path, 6)
    cores = physical_cores(range(12), cpu_dir)
    assert cores == ((0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11))
    pools = split(cores)
    assert pools.cpu_slots == ((0, 1), (2, 3), (4, 5))
    assert pools.gpu_threads == (6, 7, 8, 9)
    assert pools.reserved_threads == (10, 11)
    assert pools.memory_budget_gb == 52.0
    assert split(cores, cpu_jobs=2).cpu_slots == ((0, 1), (2, 3))
    narrowed = physical_cores([0, 1, 2, 3, 6, 7], cpu_dir)
    assert narrowed == ((0, 6), (1, 7), (2,), (3,))
    pools = split(narrowed)
    assert pools.cpu_slots == ((0, 1), (2, 3))
    assert pools.reserved_threads == (6, 7)
    assert pools.gpu_threads == (0, 1, 2, 3)
    assert any("share first threads" in note for note in pools.notes)


def test_a_small_machine_still_gets_one_cpu_slot():
    pools = split(((0,), (1,)), cpu_jobs=5, cores_per_cpu_job=4)
    assert pools.cpu_slots == ((0, 1),)
    assert pools.reserved_threads == ()
    assert len(pools.notes) == 3


def test_second_threads_become_cpu_slots_after_the_first_threads_only_when_asked(tmp_path):
    cores = physical_cores(range(14), topology(tmp_path, 7))
    pools = split(cores, cpu_jobs_on_second_threads=True)
    assert pools.cpu_slots == ((0, 1), (2, 3), (4, 5), (7, 8), (9, 10))
    assert pools.gpu_threads == (11,)
    assert pools.reserved_threads == (12, 13)
    assert pools.shared_cores
    assert split(cores, cpu_jobs=4, cpu_jobs_on_second_threads=True).cpu_slots[3:] == ((7, 8),)
    assert split(cores, cpu_jobs=3, cpu_jobs_on_second_threads=True).shared_cores is False
    assert split(cores).cpu_slots == ((0, 1), (2, 3), (4, 5))
    assert split(((0,), (1,)), cpu_jobs_on_second_threads=True).shared_cores is False


def test_shared_cpu_slots_leave_a_second_thread_for_every_gpu_worker(tmp_path):
    cores = physical_cores(range(20), topology(tmp_path, 10))
    resources = RESOURCES.model_copy(update={"cpu_jobs_on_second_threads": True})

    def pools(devices):
        gpus = tuple(Gpu(k, f"GPU-{k}", "card", 16.0, "8.6") for k in range(devices))
        hardware = Hardware(cores=cores, memory_gb=64.0, available_memory_gb=60.0, gpus=gpus)
        return allocate(hardware, resources, cpu_jobs=None, reserved_memory_gb=8.0, gpus=gpus)

    assert pools(2).cpu_slots[5:] == ((10, 11), (12, 13))
    assert pools(2).gpu_threads == (14, 15, 16, 17)
    assert pools(2).reserved_threads == (18, 19)
    assert pools(4).shared_cores is False
    assert pools(4).gpu_threads == tuple(range(10, 18))


def test_gpu_workers_can_keep_whole_cores_beside_the_cpu_slots(tmp_path):
    cores = physical_cores(range(20), topology(tmp_path, 10))
    gpus = tuple(Gpu(k, f"GPU-{k}", "card", 16.0, "8.6") for k in range(2))
    hardware = Hardware(cores=cores, memory_gb=64.0, available_memory_gb=60.0, gpus=gpus)

    def pools(**changes):
        resources = RESOURCES.model_copy(update={"gpu_workers_keep_whole_cores": True, **changes})
        return allocate(hardware, resources, cpu_jobs=None, reserved_memory_gb=8.0, gpus=gpus)

    assert pools().cpu_slots == ((0, 1), (2, 3), (4, 5))
    assert pools().gpu_threads == (6, 7, 8, 9, 16, 17)
    assert pools().reserved_threads == (18, 19)
    shared = pools(cpu_jobs_on_second_threads=True)
    assert shared.cpu_slots[3:] == ((10, 11), (12, 13), (14, 15))
    assert shared.gpu_threads == (6, 7, 8, 9, 16, 17)
    crowded = pools(gpu_workers_per_device=5)
    assert crowded.gpu_threads == tuple(range(10, 18))
    assert any("too few" in note for note in crowded.notes)


def test_cuda_visible_devices_picks_devices_as_cuda_does():
    gpus = [
        Gpu(0, "GPU-aaa1", "card", 16.0, "8.6"),
        Gpu(1, "GPU-bbb2", "card", 16.0, "8.6"),
        Gpu(3, "GPU-ddd4", "card", 16.0, "8.6"),
    ]
    assert visible(gpus, None) == tuple(gpus)
    assert visible(gpus, "3,0") == (gpus[2], gpus[0])
    assert visible(gpus, "GPU-bbb") == (gpus[1],)
    assert visible(gpus, "1,7,0") == (gpus[1],)
    assert visible(gpus, "") == ()
    assert visible(gpus, "-1") == ()


def test_memory_reads_kilobytes_as_gb(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        65755432 kB\nMemFree:         40271144 kB\nMemAvailable:    61204716 kB\n"
    )
    total, available = memory(meminfo, tmp_path / "cgroup", tmp_path / "self")
    assert total == pytest.approx(62.7093, abs=1e-4)
    assert available == pytest.approx(58.3694, abs=1e-4)


def test_cgroup_memory_limits_cap_the_memory_less_what_their_cgroup_uses(tmp_path):
    gb = 2**30
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: {64 * 2**20} kB\nMemAvailable: {60 * 2**20} kB\n")
    root = tmp_path / "cgroup"
    leaf = root / "system.slice" / "job.scope"
    leaf.mkdir(parents=True)
    membership = tmp_path / "self"
    membership.write_text("0::/system.slice/job.scope\n")
    assert memory(meminfo, root, membership) == (64.0, 60.0)
    (root / "system.slice" / "memory.max").write_text("max\n")
    (leaf / "memory.max").write_text(f"{16 * gb}\n")
    (leaf / "memory.current").write_text(f"{10 * gb}\n")
    (leaf / "memory.stat").write_text(f"active_file {gb}\ninactive_file {3 * gb}\n")
    assert memory(meminfo, root, membership) == (16.0, 9.0)
    (root / "system.slice" / "memory.max").write_text(f"{12 * gb}\n")
    (root / "system.slice" / "memory.current").write_text(f"{11 * gb}\n")
    assert memory(meminfo, root, membership) == (12.0, 1.0)


def test_cgroup_v1_memory_limits_follow_the_process_below_the_mounted_top(tmp_path):
    gb = 2**30
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: {64 * 2**20} kB\nMemAvailable: {60 * 2**20} kB\n")
    point = tmp_path / "v1" / "memory"
    leaf = point / "system.slice" / "job.scope"
    leaf.mkdir(parents=True)
    proc = tmp_path / "proc"
    proc.mkdir()
    membership = proc / "cgroup"
    membership.write_text("4:cpu,cpuacct:/system.slice\n7:memory:/system.slice/job.scope\n")
    mounts = proc / "mountinfo"
    mounts.write_text(f"33 24 0:28 / {point} rw,nosuid shared:15 - cgroup cgroup rw,memory\n")
    (point / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (leaf / "memory.limit_in_bytes").write_text(f"{8 * gb}\n")
    (leaf / "memory.usage_in_bytes").write_text(f"{6 * gb}\n")
    assert memory(meminfo, tmp_path / "v2", membership) == (8.0, 2.0)
    (point / "system.slice" / "memory.limit_in_bytes").write_text(f"{7 * gb}\n")
    (point / "system.slice" / "memory.usage_in_bytes").write_text(f"{13 * gb // 2}\n")
    assert memory(meminfo, tmp_path / "v2", membership) == (7.0, 0.5)
    (leaf / "memory.stat").write_text(f"inactive_file 0\ntotal_inactive_file {2 * gb}\n")
    container = f"33 24 0:28 /system.slice/job.scope {leaf} rw - cgroup cgroup rw,memory\n"
    mounts.write_text(container)
    assert memory(meminfo, tmp_path / "v2", membership) == (8.0, 4.0)


def test_a_cgroup_cpu_quota_caps_the_cores_used(tmp_path):
    root = tmp_path / "cgroup"
    leaf = root / "system.slice" / "job.scope"
    leaf.mkdir(parents=True)
    membership = tmp_path / "self"
    membership.write_text("0::/system.slice/job.scope\n")
    (root / "system.slice" / "cpu.max").write_text("max 100000\n")
    (leaf / "cpu.max").write_text("250000 100000\n")
    assert cpu_quota(root, membership) == 2.5
    (root / "system.slice" / "cpu.max").write_text("150000 100000\n")
    assert cpu_quota(root, membership) == 1.5
    (leaf / "cpu.max").write_text("max 100000\n")
    (root / "system.slice" / "cpu.max").write_text("max 100000\n")
    assert cpu_quota(root, membership) is None
    point = tmp_path / "v1" / "cpu"
    (point / "system.slice").mkdir(parents=True)
    membership.write_text("4:cpu,cpuacct:/system.slice\n0::/system.slice/job.scope\n")
    mounts = f"30 24 0:27 / {point} rw - cgroup cgroup rw,cpu,cpuacct\n"
    (tmp_path / "mountinfo").write_text(mounts)
    (point / "cpu.cfs_quota_us").write_text("-1\n")
    (point / "system.slice" / "cpu.cfs_quota_us").write_text("300000\n")
    (point / "system.slice" / "cpu.cfs_period_us").write_text("100000\n")
    assert cpu_quota(root, membership) == 3.0
    cores = tuple((k, k + 6) for k in range(6))
    hardware = Hardware(cores, 64.0, 60.0, (), cpu_quota=2.5)
    pools = allocate(hardware, RESOURCES, cpu_jobs=None, reserved_memory_gb=8.0, gpus=())
    assert pools.cpu_slots == ((0, 1),)
    assert pools.reserved_threads == (6, 7)
    assert "a cgroup cpu quota of 2.5 cpus leaves 2 of 6 cores" in pools.notes
