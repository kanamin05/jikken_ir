"""One-way DQN -> guarded route handoff for the fixed Task2A map."""
import numpy as np


class HandoffRequested(RuntimeError):
    pass


def handoff_reason(position, scoring, exit_y=None):
    """Use new score, not a score left over from a previous run.

    This course proceeds in map +Y. An optional surveyed exit plane must be
    behind the obstacles including the robot footprint; raw surfaces include
    walls and cannot safely provide that plane automatically.
    """
    if not np.isfinite(position).all():
        return None
    if exit_y is not None and position[1] >= exit_y:
        return 'passed_obstacle_exit_y'
    initial, latest = scoring.get('score_start'), scoring.get('score_latest')
    if (initial is not None and latest is not None and latest > initial and
            (scoring.get('room2_count') or 0) > 0 and position[1] >= 2.2):
        return 'new_score_and_room2_near_obstacles'
    return None
