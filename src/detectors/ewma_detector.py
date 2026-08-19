"""Exponentially weighted moving average anomaly detection."""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeBaseline

logger = logging.getLogger(__name__)


class EWMADetector:
    """Detects anomalies using an exponentially weighted moving average."""

    def __init__(
        self,
        alpha: float = 0.3,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
    ) -> None:
        self.alpha = alpha
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._ewma_values: dict[str, float] = {}

    def detect(
        self,
        observation: TransactionVolume,
        baseline: VolumeBaseline,
    ) -> Optional[AnomalyEvent]:
        """Check a single observation against its EWMA."""
        if baseline.std_count == 0:
            return None

        key = f"{observation.service_name}/{observation.endpoint}"
        ewma = self._ewma_values.get(key, baseline.mean_count)
        deviation = (observation.count - ewma) / baseline.std_count
        abs_deviation = abs(deviation)

        self._ewma_values[key] = (
            self.alpha * observation.count + (1 - self.alpha) * ewma
        )

        if abs_deviation < self.warning_threshold:
            return None

        severity = self._classify_severity(abs_deviation)
        anomaly_type = (
            AnomalyType.VOLUME_SPIKE if deviation > 0 else AnomalyType.VOLUME_DROP
        )

        direction = "above" if deviation > 0 else "below"
        description = (
            f"{observation.service_name}/{observation.endpoint}: "
            f"volume {observation.count} is {abs_deviation:.1f} std devs {direction} "
            f"expected {ewma:.0f} "
            f"(hour={baseline.hour_of_day}, dow={baseline.day_of_week})"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-{uuid4().hex[:8]}",
            anomaly_type=anomaly_type,
            severity=severity,
            service_name=observation.service_name,
            endpoint=observation.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=float(observation.count),
            expected_value=ewma,
            deviation_score=abs_deviation,
            description=description,
        )

    def _classify_severity(self, abs_deviation: float) -> AnomalySeverity:
        """Map a deviation magnitude to a severity level."""
        if abs_deviation >= self.critical_threshold * 2:
            return AnomalySeverity.CRITICAL
        if abs_deviation >= self.critical_threshold:
            return AnomalySeverity.HIGH
        if abs_deviation >= self.warning_threshold:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW
