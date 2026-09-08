"""明确失败，不用 reference 偷偷代替学生实现。"""
class ExerciseNotImplemented(NotImplementedError):
    pass


def todo(task_id: str):
    raise ExerciseNotImplemented(
        f"{task_id} 尚未完成；见 docs/TASKS.md。实现后移除对应 todo 占位调用。"
    )
