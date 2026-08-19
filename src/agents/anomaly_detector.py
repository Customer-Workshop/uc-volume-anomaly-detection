"""Anomaly Detection Agent — establishes baselines and identifies volume deviations."""

import argparse
import csv
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.detectors.seasonal_detector import SeasonalDetector
from src.detectors.zscore_detector import ZScoreDetector
from src.models.anomaly import AnomalyEvent
from src.models.transaction import TransactionVolume, VolumeBaseline, VolumeTimeSeries

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent:
    """Orchestrates anomaly detection using multiple detection algorithms."""

    def __init__(
        self,
        zscore_warning: float = 2.0,
        zscore_critical: float = 3.0,
        seasonal_threshold: float = 2.5,
    ) -> None:
        self.zscore_detector = ZScoreDetector(
            warning_threshold=zscore_warning,
            critical_threshold=zscore_critical,
        )
        self.seasonal_detector = SeasonalDetector(
            deviation_threshold=seasonal_threshold,
        )
        self.baselines: dict[str, list[VolumeBaseline]] = {}
        self.detected_anomalies: list[AnomalyEvent] = []

    def load_historical_data(self, csv_path: str) -> dict[str, VolumeTimeSeries]:
        """Load historical transaction data from CSV."""
        path = Path(csv_path)
        if not path.exists():
            logger.error("Historical data file not found: %s", csv_path)
            return {}

        series_map: dict[str, VolumeTimeSeries] = {}

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['service_name']}/{row['endpoint']}"
                if key not in series_map:
                    series_map[key] = VolumeTimeSeries(
                        service_name=row["service_name"],
                        endpoint=row["endpoint"],
                    )

                obs = TransactionVolume(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    service_name=row["service_name"],
                    endpoint=row["endpoint"],
                    count=int(row["count"]),
                    error_count=int(row.get("error_count", "0")),
                    avg_latency_ms=float(row.get("avg_latency_ms", "0")),
                    p99_latency_ms=float(row.get("p99_latency_ms", "0")),
                )
                series_map[key].observations.append(obs)

        logger.info("Loaded %d time series from %s", len(series_map), csv_path)
        return series_map

    def build_baselines(
        self, historical_data: dict[str, VolumeTimeSeries]
    ) -> None:
        """Build seasonal baselines from historical data."""
        for key, series in historical_data.items():
            baselines = self.seasonal_detector.build_baselines(series)
            self.baselines[key] = baselines

        total = sum(len(bl) for bl in self.baselines.values())
        logger.info("Built %d baselines across %d series", total, len(self.baselines))

    def analyze(
        self, observation: TransactionVolume
    ) -> list[AnomalyEvent]:
        """Analyze a single observation for anomalies using all detectors."""
        key = f"{observation.service_name}/{observation.endpoint}"
        anomalies: list[AnomalyEvent] = []

        baselines = self.baselines.get(key, [])
        if not baselines:
            logger.debug("No baselines for %s, skipping", key)
            return anomalies

        # Seasonal detection
        seasonal_anomaly = self.seasonal_detector.detect(observation, baselines)
        if seasonal_anomaly:
            anomalies.append(seasonal_anomaly)

        # Z-score detection against matching baseline
        hour = observation.timestamp.hour
        dow = observation.timestamp.weekday()
        matching_baseline = next(
            (b for b in baselines if b.hour_of_day == hour and b.day_of_week == dow),
            None,
        )
        if matching_baseline:
            zscore_anomaly = self.zscore_detector.detect(observation, matching_baseline)
            if zscore_anomaly:
                anomalies.append(zscore_anomaly)

            latency_anomaly = self.zscore_detector.detect_latency(
                observation, matching_baseline
            )
            if latency_anomaly:
                anomalies.append(latency_anomaly)

        self.detected_anomalies.extend(anomalies)
        return anomalies


def main() -> int:
    """Run batch anomaly detection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/historical/sample_transactions.csv")
    parser.add_argument("--mode", choices=("batch", "live"), default="batch")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=20,
        metavar="N",
        help="maximum number of anomaly details to print (default: 20)",
    )
    args = parser.parse_args()
    if args.max_details < 0:
        parser.error("--max-details must be non-negative")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if args.mode == "live":
        print(
            "Live mode requires a METRICS_ENDPOINT metrics client, "
            "which is not implemented yet.",
            file=sys.stderr,
        )
        return 1

    agent = AnomalyDetectionAgent()
    historical_data = agent.load_historical_data(args.data)
    if not historical_data:
        print(f"No historical data loaded from {args.data}.", file=sys.stderr)
        return 1

    agent.build_baselines(historical_data)
    if not any(agent.baselines.values()):
        print(
            "No anomalies detected because no seasonal baselines were built. "
            "Generate more data with "
            "`python -m src.utils.generate_sample_data`."
        )
        return 0

    anomalies_by_series: Counter[str] = Counter()
    detected: list[tuple[datetime, AnomalyEvent]] = []
    for key, series in historical_data.items():
        for observation in series.observations:
            for anomaly in agent.analyze(observation):
                anomalies_by_series[key] += 1
                detected.append((observation.timestamp, anomaly))

    print("Detected anomalies by series:")
    for key, series in historical_data.items():
        print(
            f"  {series.service_name}{series.endpoint}: "
            f"{anomalies_by_series[key]}"
        )

    if not detected:
        print("No anomalies detected.")
        return 0

    sorted_anomalies = sorted(
        detected,
        key=lambda item: item[1].deviation_score,
        reverse=True,
    )
    details = sorted_anomalies[: args.max_details]
    print(
        f"Showing top {len(details)} of {len(detected)} anomalies "
        "by deviation score:"
    )
    for timestamp, anomaly in details:
        print(
            f"{timestamp.isoformat()} {anomaly.service_name}{anomaly.endpoint} "
            f"type={anomaly.anomaly_type.value} severity={anomaly.severity.value} "
            f"observed={anomaly.observed_value:.1f} "
            f"expected={anomaly.expected_value:.1f} "
            f"deviation={anomaly.deviation_score:.2f}"
        )

    severity_counts = Counter(anomaly.severity.value for _, anomaly in detected)
    totals = ", ".join(
        f"{severity}={severity_counts[severity]}"
        for severity in ("low", "medium", "high", "critical")
        if severity_counts[severity]
    )
    print(f"Totals by severity: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
