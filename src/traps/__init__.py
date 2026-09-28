"""TRAPs 表層短期吸引潛勢分析套件的公開數值入口。"""

from .detection import detect_snapshot
from .models import DetectionConfig, SnapshotResult, TrapDetection

__all__ = ["DetectionConfig", "SnapshotResult", "TrapDetection", "detect_snapshot"]

