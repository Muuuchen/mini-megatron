"""TODO 进度与启动前检查；移除占位符不代表实现正确，仍须跑测试。"""
import argparse
import ast
from pathlib import Path

# task -> (day, title, file, prerequisite IDs)
TASKS = {
    "T01": (1, "next-token mean CE", "training/loss.py", []),
    "T02": (2, "DP/PP/TP group plan", "distributed/topology.py", []),
    "T03": (2, "copy backward: TP SUM", "tensor_parallel/mappings.py", ["T02"]),
    "T04": (2, "reduce forward: TP SUM", "tensor_parallel/mappings.py", ["T02"]),
    "T05": (2, "column-parallel linear", "tensor_parallel/layers.py", ["T03"]),
    "T06": (2, "row-parallel linear", "tensor_parallel/layers.py", ["T04"]),
    "T07": (2, "vocab-parallel embedding", "tensor_parallel/layers.py", ["T04"]),
    "T08": (4, "vocab CE forward", "tensor_parallel/loss.py", ["T02", "T01"]),
    "T09": (4, "vocab CE backward", "tensor_parallel/loss.py", ["T08"]),
    "T10": (3, "head-parallel attention", "model/parallel.py", ["T05", "T06"]),
    "T11": (3, "tensor-parallel MLP", "model/parallel.py", ["T05", "T06"]),
    "T12": (3, "pre-norm residual block", "model/parallel.py", ["T10", "T11"]),
    "T13": (3, "pipeline-local GPT stage", "model/parallel.py", ["T07", "T12"]),
    "T14": (4, "DP gradient averaging", "training/grad.py", ["T02", "T15"]),
    "T15": (3, "microbatch accumulation", "training/schedules.py", ["T01", "T13"]),
    "T16": (5, "GPipe forward and graph cache", "pipeline_parallel/schedules.py", ["T13", "T15"]),
    "T17": (5, "GPipe reverse backward", "pipeline_parallel/schedules.py", ["T16"]),
    "T18": (6, "TP/PP-aware grad norm", "training/grad.py", ["T14", "T17"]),
    "T19": (6, "same-topology checkpoint save", "checkpoint.py", ["T15"]),
    "T20": (6, "same-topology checkpoint restore", "checkpoint.py", ["T19"]),
}


from megatron_mini.labs.manifest import EXTRA_TASKS
TASKS.update(EXTRA_TASKS)
# v2压缩主线：不再沿用v1每天只做少量TP/DP任务的日程。
for _day, _ids in {1: [1,2,3,4,5,6,7,10,11,12,13,15], 2: [8,9,14],
                   4: [16,17], 5: [18], 7: [19,20]}.items():
    for _i in _ids:
        _key=f"T{_i:02}"
        TASKS[_key]=(_day,*TASKS[_key][1:])


def pending_tasks():
    found = {}
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "todo" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                task = node.args[0].value
                found[task] = f"{path.relative_to(root)}:{node.lineno}"
    return found


def require_tasks(ids):
    unknown = set(ids) - TASKS.keys()
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    pending = pending_tasks()
    missing = sorted(set(ids) & pending.keys())
    if missing:
        details = "\n".join(f"  {t}: {TASKS[t][1]} ({pending[t]})" for t in missing)
        raise SystemExit("所需练习尚未实现，尚未启动分布式通信：\n" + details
                         + "\n见 docs/TASKS.md；不要仅删除占位符，须通过对应验收。")


def model_tasks(tp=1):
    return ["T03", "T04", "T05", "T06", "T07", "T10", "T11", "T12", "T13"]


def training_tasks(dims, clip=False, save=False, resume=False):
    ids = model_tasks(dims.tp)
    ids += ["T01"] if dims.tp == 1 else ["T08", "T09"]
    ids += ["T15"] if dims.pp == 1 else ["T16", "T17"]
    if dims.world_size > 1: ids += ["T02"]
    if dims.dp > 1: ids += ["T14"]
    if clip: ids += ["T18"]
    if save: ids += ["T19"]
    if resume: ids += ["T20"]
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", nargs="*", help="fail if specified placeholders remain")
    args = parser.parse_args()
    if args.check is not None:
        require_tasks(args.check)
        print("占位符检查通过；继续运行数值验收。")
        return
    pending = pending_tasks()
    for task, (day, title, path, deps) in sorted(TASKS.items(), key=lambda kv:(kv[1][0],kv[0])):
        status = "TODO" if task in pending else "待验收"
        print(f"D{day}  {task}  {status:4}  {title:38}  {pending.get(task, path)}")
    print(f"\n{len(pending)}/{len(TASKS)} 个占位符未完成。无占位符不等于数值正确。")


if __name__ == "__main__":
    main()
