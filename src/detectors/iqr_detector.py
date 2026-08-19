"""Interquartile range (IQR) based anomaly detection for transaction volumes."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class IQRBounds:
    """Quartile-based bounds for a service endpoint's transaction volume."""

    service_name: str
    endpoint: str
    q1: float
    q3: float
    median: float
    sample_size: int = 0

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


class IQRDetector:
    """Detects anomalies by measuring how far an observation falls outside the IQR.

    Unlike z-score detection, quartile-based fences are robust to the outliers
    present in the historical window itself, so a single past incident does not
    inflate the expected range.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        critical_multiplier: float = 3.0,
        min_samples: int = 4,
    ) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.critical_multiplier = critical_multiplier
        self.min_samples = min_samples

    def build_bounds(self, time_series: VolumeTimeSeries) -> Optional[IQRBounds]:
        """Compute quartile bounds from a series of historical observations."""
        counts = [float(obs.count) for obs in time_series.observations]
        if len(counts) < self.min_samples:
            return None

        return IQRBounds(
            service_name=time_series.service_name,
            endpoint=time_series.endpoint,
            q1=_percentile(counts, 0.25),
            q3=_percentile(counts, 0.75),
            median=_percentile(counts, 0.5),
            sample_size=len(counts),
        )

    def detect(
        self,
        observation: TransactionVolume,
        baseline: IQRBounds,
    ) -> Optional[AnomalyEvent]:
        """Check a single observation against its quartile bounds."""
        iqr = baseline.iqr
        if iqr == 0:
            return None

        if observation.count > baseline.q3:
            excess = (observation.count - baseline.q3) / iqr
            anomaly_type = AnomalyType.VOLUME_SPIKE
            direction = "above"
            quartile = "upper"
            fence = baseline.q3 + self.iqr_multiplier * iqr
        elif observation.count < baseline.q1:
            excess = (baseline.q1 - observation.count) / iqr
            anomaly_type = AnomalyType.VOLUME_DROP
            direction = "below"
            quartile = "lower"
            fence = max(0.0, baseline.q1 - self.iqr_multiplier * iqr)
        else:
            return None

        if excess < self.iqr_multiplier:
            return None

        severity = self._classify_severity(excess)
        description = (
            f"IQR anomaly: {observation.service_name}/{observation.endpoint} "
            f"volume={observation.count} is {excess:.1f}×IQR {direction} the "
            f"{quartile} quartile (fence={fence:.0f}, "
            f"median={baseline.median:.0f}, samples={baseline.sample_size})"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-{uuid4().hex[:8]}",
            anomaly_type=anomaly_type,
            severity=severity,
            service_name=observation.service_name,
            endpoint=observation.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=float(observation.count),
            expected_value=baseline.median,
            deviation_score=excess,
            description=description,
        )

    def _classify_severity(self, excess: float) -> AnomalySeverity:
        """Map an IQR-multiple magnitude to a severity level."""
        if excess >= self.critical_multiplier * 2:
            return AnomalySeverity.CRITICAL
        if excess >= self.critical_multiplier:
            return AnomalySeverity.HIGH
        if excess >= self.iqr_multiplier:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile of an unsorted list of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
