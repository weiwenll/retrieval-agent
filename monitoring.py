"""
LLM Monitoring and Metrics Collection
Tracks OpenAI API calls, latency, token usage, and costs for Grafana export
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetric:
    """Single LLM API call metric"""
    timestamp: str
    model: str
    operation: str  # e.g., "tag_generation", "interest_mapping", "description_creation"
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    success: bool
    error: Optional[str] = None


class LLMMonitor:
    """
    Monitor and track all LLM API calls for metrics export.

    Tracks:
    - Latency per call
    - Token usage (input/output)
    - Cost estimation
    - Success/failure rates
    - Response times by operation type

    Export formats:
    - JSON (for Grafana JSON datasource)
    - Prometheus format
    - Summary statistics
    """

    # OpenAI pricing (as of 2024) - update as needed
    PRICING = {
        "gpt-4o": {
            "input": 0.0025 / 1000,   # $2.50 per 1M input tokens
            "output": 0.010 / 1000     # $10.00 per 1M output tokens
        },
        "gpt-4o-mini": {
            "input": 0.00015 / 1000,   # $0.15 per 1M input tokens
            "output": 0.0006 / 1000    # $0.60 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 0.01 / 1000,
            "output": 0.03 / 1000
        },
        "gpt-3.5-turbo": {
            "input": 0.0005 / 1000,
            "output": 0.0015 / 1000
        }
    }

    def __init__(self):
        self.metrics: List[LLMCallMetric] = []
        self.start_time = time.time()

        # Aggregated stats
        self.total_calls = 0
        self.total_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0,
            "total_tokens": 0,
            "errors": 0
        })

    def track_call(
        self,
        model: str,
        operation: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Track a single LLM API call"""
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        metric = LLMCallMetric(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            success=success,
            error=error
        )

        self.metrics.append(metric)

        # Update aggregated stats
        self.total_calls += 1
        if not success:
            self.total_errors += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        # Update operation stats
        op_stats = self.operation_stats[operation]
        op_stats["count"] += 1
        op_stats["total_latency_ms"] += latency_ms
        op_stats["total_tokens"] += total_tokens
        if not success:
            op_stats["errors"] += 1

        logger.info(
            f"LLM Call [{operation}] - Model: {model}, "
            f"Latency: {latency_ms:.0f}ms, Tokens: {total_tokens} "
            f"({input_tokens} in + {output_tokens} out), Cost: ${cost:.4f}"
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing"""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        elapsed_time = time.time() - self.start_time

        summary = {
            "overview": {
                "total_calls": self.total_calls,
                "total_errors": self.total_errors,
                "success_rate": (self.total_calls - self.total_errors) / max(self.total_calls, 1),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "calls_per_minute": round(self.total_calls / max(elapsed_time / 60, 0.01), 2)
            },
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
                "avg_per_call": round((self.total_input_tokens + self.total_output_tokens) / max(self.total_calls, 1), 2)
            },
            "cost": {
                "total_usd": round(self.total_cost_usd, 4),
                "avg_per_call_usd": round(self.total_cost_usd / max(self.total_calls, 1), 4)
            },
            "latency": {
                "avg_ms": round(sum(m.latency_ms for m in self.metrics) / max(len(self.metrics), 1), 2),
                "min_ms": round(min((m.latency_ms for m in self.metrics), default=0), 2),
                "max_ms": round(max((m.latency_ms for m in self.metrics), default=0), 2),
                "p95_ms": self._calculate_percentile([m.latency_ms for m in self.metrics], 95)
            },
            "by_operation": {}
        }

        # Per-operation breakdown
        for operation, stats in self.operation_stats.items():
            summary["by_operation"][operation] = {
                "count": stats["count"],
                "avg_latency_ms": round(stats["total_latency_ms"] / max(stats["count"], 1), 2),
                "total_tokens": stats["total_tokens"],
                "avg_tokens": round(stats["total_tokens"] / max(stats["count"], 1), 2),
                "errors": stats["errors"],
                "success_rate": (stats["count"] - stats["errors"]) / max(stats["count"], 1)
            }

        return summary

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return round(sorted_values[min(index, len(sorted_values) - 1)], 2)

    def export_grafana_json(self, output_file: str = "metrics/llm_metrics.json"):
        """Export metrics in Grafana JSON format"""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_summary(),
            "metrics": [asdict(m) for m in self.metrics]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(self.metrics)} LLM metrics to {output_file}")
        return output_file

    def export_prometheus(self, output_file: str = "metrics/llm_metrics.prom"):
        """Export metrics in Prometheus format"""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        lines = []

        # Total calls
        lines.append("# HELP llm_calls_total Total number of LLM API calls")
        lines.append("# TYPE llm_calls_total counter")
        lines.append(f"llm_calls_total {self.total_calls}")

        # Errors
        lines.append("# HELP llm_errors_total Total number of LLM API errors")
        lines.append("# TYPE llm_errors_total counter")
        lines.append(f"llm_errors_total {self.total_errors}")

        # Tokens
        lines.append("# HELP llm_tokens_input_total Total input tokens")
        lines.append("# TYPE llm_tokens_input_total counter")
        lines.append(f"llm_tokens_input_total {self.total_input_tokens}")

        lines.append("# HELP llm_tokens_output_total Total output tokens")
        lines.append("# TYPE llm_tokens_output_total counter")
        lines.append(f"llm_tokens_output_total {self.total_output_tokens}")

        # Cost
        lines.append("# HELP llm_cost_usd_total Total cost in USD")
        lines.append("# TYPE llm_cost_usd_total counter")
        lines.append(f"llm_cost_usd_total {self.total_cost_usd}")

        # Latency histogram
        lines.append("# HELP llm_latency_ms LLM call latency in milliseconds")
        lines.append("# TYPE llm_latency_ms histogram")
        for metric in self.metrics:
            lines.append(
                f'llm_latency_ms{{model="{metric.model}",operation="{metric.operation}"}} '
                f'{metric.latency_ms}'
            )

        # Per-operation metrics
        for operation, stats in self.operation_stats.items():
            lines.append(f'# HELP llm_operation_calls_total Calls by operation type')
            lines.append(f'# TYPE llm_operation_calls_total counter')
            lines.append(f'llm_operation_calls_total{{operation="{operation}"}} {stats["count"]}')

        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported Prometheus metrics to {output_file}")
        return output_file

    def print_summary(self):
        """Print formatted summary to console"""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("LLM METRICS SUMMARY")
        print("="*80)

        print(f"\nOverview:")
        print(f"  Total Calls:     {summary['overview']['total_calls']}")
        print(f"  Errors:          {summary['overview']['total_errors']}")
        print(f"  Success Rate:    {summary['overview']['success_rate']:.1%}")
        print(f"  Elapsed Time:    {summary['overview']['elapsed_time_seconds']:.2f}s")
        print(f"  Calls/Minute:    {summary['overview']['calls_per_minute']:.1f}")

        print(f"\nToken Usage:")
        print(f"  Input Tokens:    {summary['tokens']['total_input']:,}")
        print(f"  Output Tokens:   {summary['tokens']['total_output']:,}")
        print(f"  Total Tokens:    {summary['tokens']['total']:,}")
        print(f"  Avg/Call:        {summary['tokens']['avg_per_call']:.0f}")

        print(f"\nCost:")
        print(f"  Total Cost:      ${summary['cost']['total_usd']:.4f}")
        print(f"  Avg/Call:        ${summary['cost']['avg_per_call_usd']:.4f}")

        print(f"\nLatency:")
        print(f"  Average:         {summary['latency']['avg_ms']:.0f}ms")
        print(f"  Min:             {summary['latency']['min_ms']:.0f}ms")
        print(f"  Max:             {summary['latency']['max_ms']:.0f}ms")
        print(f"  P95:             {summary['latency']['p95_ms']:.0f}ms")

        if summary['by_operation']:
            print(f"\nBy Operation:")
            for operation, stats in summary['by_operation'].items():
                print(f"\n  {operation}:")
                print(f"    Calls:         {stats['count']}")
                print(f"    Avg Latency:   {stats['avg_latency_ms']:.0f}ms")
                print(f"    Avg Tokens:    {stats['avg_tokens']:.0f}")
                print(f"    Success Rate:  {stats['success_rate']:.1%}")

        print("\n" + "="*80 + "\n")


class MonitoredLLMClient:
    """
    Wrapper for OpenAI client that automatically tracks all calls.
    Drop-in replacement for openai.OpenAI client.
    """

    def __init__(self, client, monitor: LLMMonitor, default_operation: str = "general"):
        self.client = client
        self.monitor = monitor
        self.default_operation = default_operation

    @property
    def chat(self):
        """Return monitored chat interface"""
        return MonitoredChat(self.client.chat, self.monitor, self.default_operation)


class MonitoredChat:
    """Monitored chat completions interface"""

    def __init__(self, chat, monitor: LLMMonitor, default_operation: str):
        self._chat = chat
        self.monitor = monitor
        self.default_operation = default_operation

    @property
    def completions(self):
        """Return monitored completions interface"""
        return MonitoredCompletions(self._chat.completions, self.monitor, self.default_operation)


class MonitoredCompletions:
    """Monitored completions interface"""

    def __init__(self, completions, monitor: LLMMonitor, default_operation: str):
        self._completions = completions
        self.monitor = monitor
        self.default_operation = default_operation

    def create(self, *args, operation: str = None, **kwargs):
        """Create completion with monitoring"""
        operation = operation or self.default_operation
        model = kwargs.get("model", "unknown")

        start_time = time.time()
        success = True
        error = None
        response = None

        try:
            response = self._completions.create(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            self.monitor.track_call(
                model=model,
                operation=operation,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True
            )

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error = str(e)
            success = False

            self.monitor.track_call(
                model=model,
                operation=operation,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=error
            )

            raise


# Global monitor instance
_global_monitor = None


def get_global_monitor() -> LLMMonitor:
    """Get or create global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = LLMMonitor()
    return _global_monitor


def reset_global_monitor():
    """Reset global monitor (useful for testing)"""
    global _global_monitor
    _global_monitor = LLMMonitor()
