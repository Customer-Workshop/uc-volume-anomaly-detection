"""Generate realistic historical transaction volume sample data."""

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

CSV_HEADER = [
    "timestamp",
    "service_name",
    "endpoint",
    "count",
    "error_count",
    "avg_latency_ms",
    "p99_latency_ms",
]
DEFAULT_CONFIG_PATH = Path("config/detection_rules.yaml")
DEFAULT_OUTPUT_PATH = Path("data/historical/sample_transactions.csv")
DEFAULT_ENDPOINTS = [("sample-service", "/api/sample")]


@dataclass(frozen=True)
class SampleEndpoint:
    """A service endpoint for which sample data should be generated."""

    service_name: str
    endpoint: str


def _load_endpoints(config_path: Path = DEFAULT_CONFIG_PATH) -> list[SampleEndpoint]:
    """Read monitored endpoints from the detection rules configuration."""
    try:
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read %s: %s", config_path, exc)
        config = {}
    if not isinstance(config, dict):
        config = {}

    services = config.get("services", {})
    monitored = services.get("monitored", []) if isinstance(services, dict) else []
    endpoints: list[SampleEndpoint] = []
    if isinstance(monitored, list):
        for service in monitored:
            if not isinstance(service, dict):
                continue
            name = service.get("name")
            service_endpoints = service.get("endpoints", [])
            if not name or not isinstance(service_endpoints, list):
                continue
            endpoints.extend(
                SampleEndpoint(str(name), str(endpoint))
                for endpoint in service_endpoints
                if endpoint
            )

    return endpoints or [
        SampleEndpoint(service_name, endpoint)
        for service_name, endpoint in DEFAULT_ENDPOINTS
    ]


def _hourly_shape(timestamp: datetime) -> float:
    """Return a weekday and diurnal multiplier for an observation."""
    hour = timestamp.hour
    if 8 <= hour < 18:
        diurnal = 1.0
    elif 6 <= hour < 8 or 18 <= hour < 22:
        diurnal = 0.55
    else:
        diurnal = 0.2
    weekday = 1.0 if timestamp.weekday() < 5 else 0.55
    return diurnal * weekday


def _base_count(endpoint: SampleEndpoint) -> float:
    """Return a stable service and endpoint-specific traffic level."""
    service_levels = {
        "payment-service": 900,
        "order-service": 780,
        "auth-service": 620,
        "notification-service": 540,
    }
    level = service_levels.get(endpoint.service_name, 500)
    if endpoint.endpoint.endswith("/status"):
        level *= 0.7
    elif endpoint.endpoint.endswith("/send"):
        level *= 0.85
    return level


def generate_sample_data(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    days: int = 90,
    seed: int = 42,
    end_time: Optional[datetime] = None,
) -> tuple[int, datetime, datetime]:
    """Generate hourly observations and write them to ``output_path``."""
    if days <= 0:
        raise ValueError("days must be greater than zero")

    rng = random.Random(seed)
    end_time = end_time or datetime(2026, 4, 1)
    start_time = end_time - timedelta(days=days)
    endpoints = _load_endpoints()
    bucket_noise: dict[tuple[str, str, int, int], tuple[float, float]] = {}
    rows: list[list[object]] = []

    for hour_index in range(days * 24):
        timestamp = start_time + timedelta(hours=hour_index)
        for endpoint in endpoints:
            expected = max(1.0, _base_count(endpoint) * _hourly_shape(timestamp))
            bucket_key = (
                endpoint.service_name,
                endpoint.endpoint,
                timestamp.hour,
                timestamp.weekday(),
            )
            if bucket_key not in bucket_noise:
                bucket_noise[bucket_key] = (
                    rng.uniform(-0.002, 0.002),
                    rng.uniform(-0.2, 0.2),
                )
            count_noise, latency_noise = bucket_noise[bucket_key]
            count = max(1, round(expected * (1.0 + count_noise)))
            avg_latency = max(
                5.0,
                28.0
                + (count / _base_count(endpoint)) * 18.0
                + latency_noise,
            )
            error_count = max(
                0,
                round(count * (0.01 + rng.uniform(-0.001, 0.001))),
            )
            rows.append(
                [
                    timestamp.isoformat(timespec="seconds"),
                    endpoint.service_name,
                    endpoint.endpoint,
                    count,
                    error_count,
                    round(avg_latency, 1),
                    round(avg_latency * 2.2 + rng.uniform(-0.5, 0.5), 1),
                ]
            )

    # Add a small number of deterministic incidents after the normal pattern.
    if rows:
        first_endpoint = endpoints[0]
        anomaly_index = (days * 24 - 2) * len(endpoints)
        rows[anomaly_index][3] = max(1, round(float(rows[anomaly_index][3]) * 4.0))
        rows[anomaly_index][4] = max(1, round(float(rows[anomaly_index][4]) * 3.0))
        rows[anomaly_index][5] = round(float(rows[anomaly_index][5]) * 1.8, 1)
        logger.debug(
            "Injected spike for %s/%s",
            first_endpoint.service_name,
            first_endpoint.endpoint,
        )
        if len(endpoints) > 1:
            drop_index = (days * 24 - 1) * len(endpoints) + 1
            rows[drop_index][3] = max(1, round(float(rows[drop_index][3]) * 0.1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    return len(rows), start_time, end_time - timedelta(hours=1)


def main() -> int:
    """Parse command-line options and generate sample data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        rows, start_time, end_time = generate_sample_data(
            output_path=args.output,
            days=args.days,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Wrote {rows} rows to {args.output} "
        f"({start_time.isoformat()} to {end_time.isoformat()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
