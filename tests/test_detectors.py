"""Tests for anomaly detection algorithms."""

from datetime import datetime

from src.detectors.zscore_detector import ZScoreDetector
from src.detectors.seasonal_detector import SeasonalDetector, _mean, _std
from src.models.anomaly import AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeBaseline, VolumeTimeSeries


class TestZScoreDetector:
    def _make_baseline(self) -> VolumeBaseline:
        return VolumeBaseline(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            hour_of_day=12,
            day_of_week=1,
            mean_count=1000.0,
            std_count=100.0,
            mean_latency_ms=50.0,
            std_latency_ms=10.0,
            sample_size=30,
        )

    def test_no_anomaly_within_threshold(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1050,
        )
        result = detector.detect(obs, baseline)
        assert result is None

    def test_detects_volume_spike(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1500,  # 5 std devs above mean
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_SPIKE

    def test_detects_volume_drop(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=500,  # 5 std devs below mean
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_DROP

    def test_severity_classification(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        # 2.5 std devs -> MEDIUM
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1250,
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.severity == AnomalySeverity.MEDIUM

        # 4 std devs -> HIGH
        obs_high = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1400,
        )
        result_high = detector.detect(obs_high, baseline)
        assert result_high is not None
        assert result_high.severity == AnomalySeverity.HIGH


class TestSeasonalDetector:
    def test_build_baselines(self):
        detector = SeasonalDetector(min_samples=2)
        series = VolumeTimeSeries(
            service_name="test-service",
            endpoint="/api/test",
            observations=[
                TransactionVolume(
                    timestamp=datetime(2026, 1, 5, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=100,
                    avg_latency_ms=50.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 12, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=120,
                    avg_latency_ms=55.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 19, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=110,
                    avg_latency_ms=52.0,
                ),
            ],
        )
        baselines = detector.build_baselines(series)
        assert len(baselines) == 1
        assert baselines[0].hour_of_day == 10
        assert baselines[0].day_of_week == 0  # Monday


class TestStatHelpers:
    def test_mean(self):
        assert _mean([10, 20, 30]) == 20.0

    def test_std(self):
        std = _std([10, 20, 30])
        assert abs(std - 10.0) < 0.01

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_std_single(self):
        assert _std([42]) == 0.0
