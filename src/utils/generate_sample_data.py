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
    last_timestamp = end_time - timedelta(hours=1)
    spike_timestamp = min(
        last_timestamp,
        max(start_time + timedelta(hours=1), end_time - timedelta(days=5)),
    )
    drop_timestamp = min(
        last_timestamp,
        max(start_time + timedelta(hours=2), end_time - timedelta(days=3)),
    )
    incidents: dict[tuple[datetime, SampleEndpoint], tuple[float, float, float]] = {}
    if endpoints:
        incidents[(spike_timestamp, endpoints[0])] = (1.65, 2.0, 0.08)
        if len(endpoints) > 1:
            incidents[(drop_timestamp, endpoints[1])] = (0.4, 1.0, 0.01)
    rows: list[list[object]] = []

    for hour_index in range(days * 24):
        timestamp = start_time + timedelta(hours=hour_index)
        for endpoint in endpoints:
            expected = max(1.0, _base_count(endpoint) * _hourly_shape(timestamp))
            count = max(1, round(rng.gauss(expected, expected * 0.06)))
            baseline_latency = 28.0 + (count / _base_count(endpoint)) * 18.0
            avg_latency = max(
                5.0,
                rng.gauss(baseline_latency, max(1.0, baseline_latency * 0.08)),
            )
            error_rate = max(0.0, rng.gauss(0.01, 0.002))
            error_count = max(0, round(count * error_rate))
            incident = incidents.get((timestamp, endpoint))
            if incident:
                count_multiplier, latency_multiplier, incident_error_rate = incident
                count = max(1, round(count * count_multiplier))
                avg_latency = max(5.0, avg_latency * latency_multiplier)
                error_count = max(error_count, round(count * incident_error_rate))
            rows.append(
                [
                    timestamp.isoformat(timespec="seconds"),
                    endpoint.service_name,
                    endpoint.endpoint,
                    count,
                    error_count,
                    round(avg_latency, 1),
                    round(avg_latency * 2.2 + rng.gauss(0, 4.0), 1),
                ]
            )

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
