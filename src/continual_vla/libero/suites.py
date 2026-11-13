"""LIBERO task suites and the tasks' names, instructions and evaluation horizons.

A task is named "<suite>/<index>", e.g. "libero_goal/3": the index of the task in LIBERO's own order of the suite
(benchmark task order 0). Sequences learn the tasks in the order of Liu et al. (2026; their released code,
github.com/Continual-VLAs/continual-openpi, _LIBERO_*_SEQ): LIBERO's order for Spatial and Goal, alphabetical for Object
and LIBERO-10 (LIU_ORDERS).

Evaluation episodes start from fixed initial scenes. Split "test" is LIBERO's 50 fixed initial states of each task,
the scenes every published LIBERO result is evaluated on; split "val" draws new initial scenes from the task's own
placement regions with seeds VAL_SEEDS, so settings can be chosen without touching the test scenes.
"""

import os

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
# Evaluation horizon in control steps (20 Hz) after the 10 settling steps: openpi's LIBERO evaluation, which Liu et al.
# use (the longest training demonstrations: libero_spatial 193, libero_object 254, libero_goal 270, libero_10 505 steps).
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
# Liu et al.'s task orders, by instruction.
LIU_ORDERS = {
    "libero_spatial": [f"pick up the black bowl {w} and place it on the plate" for w in (
        "between the plate and the ramekin", "next to the ramekin", "from table center", "on the cookie box",
        "in the top drawer of the wooden cabinet", "on the ramekin", "next to the cookie box", "on the stove",
        "next to the plate", "on the wooden cabinet")],
    "libero_object": [f"pick up the {o} and place it in the basket" for o in (
        "alphabet soup", "bbq sauce", "butter", "chocolate pudding", "cream cheese", "ketchup", "milk", "orange juice",
        "salad dressing", "tomato sauce")],
    "libero_goal": ["open the middle drawer of the cabinet", "put the bowl on the stove",
                    "put the wine bottle on top of the cabinet", "open the top drawer and put the bowl inside",
                    "put the bowl on top of the cabinet", "push the plate to the front of the stove",
                    "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate",
                    "put the wine bottle on the rack"],
    "libero_10": ["pick up the book and place it in the back compartment of the caddy",
                  "put both moka pots on the stove",
                  "put both the alphabet soup and the cream cheese box in the basket",
                  "put both the alphabet soup and the tomato sauce in the basket",
                  "put both the cream cheese box and the butter in the basket",
                  "put the black bowl in the bottom drawer of the cabinet and close it",
                  "put the white mug on the left plate and put the yellow and white mug on the right plate",
                  "put the white mug on the plate and put the chocolate pudding to the right of the plate",
                  "put the yellow and white mug in the microwave and close it",
                  "turn on the stove and put the moka pot on it"],
}
VAL_SEEDS = list(range(100000, 100500))
N_TEST_EPISODES = 50  # LIBERO's fixed initial states per task


def _configure():
    """LIBERO reads a config file of its paths and asks for one on the console at its first import; write it under
    CVLA_ROOT with the paths of the installed package (task files, initial states, and the simulation assets that
    scripts/download_libero.py installs)."""
    import importlib.util

    path = os.environ.setdefault("LIBERO_CONFIG_PATH",
                                 os.path.join(os.environ.get("CVLA_ROOT", os.path.expanduser("~")), "libero"))
    config = os.path.join(path, "config.yaml")
    if os.path.exists(config):
        return
    root = os.path.join(importlib.util.find_spec("libero").submodule_search_locations[0], "libero")
    os.makedirs(path, exist_ok=True)
    paths = {"benchmark_root": root, "bddl_files": os.path.join(root, "bddl_files"),
             "init_states": os.path.join(root, "init_files"), "assets": os.path.join(root, "assets"),
             "datasets": os.path.join(path, "datasets")}
    with open(config + f".{os.getpid()}.tmp", "w") as f:
        f.write("".join(f"{k}: {v}\n" for k, v in paths.items()))
    os.replace(config + f".{os.getpid()}.tmp", config)


def benchmark_suite(suite):
    _configure()
    from libero.libero import benchmark

    assert suite in SUITES, f"unknown suite {suite}"
    return benchmark.get_benchmark_dict()[suite](task_order_index=0)


def suite_tasks(suite):
    return [f"{suite}/{i}" for i in range(TASKS_PER_SUITE)]


def parse(task):
    suite, index = task.split("/")
    assert suite in SUITES and 0 <= int(index) < TASKS_PER_SUITE, f"unknown task {task}"
    return suite, int(index)


def instruction(task):
    suite, index = parse(task)
    return benchmark_suite(suite).get_task(index).language


def liu_order(suite):
    """Task indices of the suite in Liu et al.'s order."""
    texts = [benchmark_suite(suite).get_task(i).language for i in range(TASKS_PER_SUITE)]
    order = [texts.index(t) for t in LIU_ORDERS[suite]]
    assert sorted(order) == list(range(TASKS_PER_SUITE)), order
    return order


def horizon(task):
    return MAX_STEPS[parse(task)[0]]
