# LLM Monitoring & Metrics

This module provides comprehensive monitoring for OpenAI API calls with Grafana export capabilities.

## Features

- **Latency Tracking**: Measure response time for each LLM call
- **Token Usage**: Track input/output tokens for cost estimation
- **Cost Calculation**: Automatic cost estimation based on model pricing
- **Operation Breakdown**: Group metrics by operation type
- **Success/Error Rates**: Monitor API reliability
- **Export Formats**: JSON (Grafana), Prometheus, Summary stats

## Quick Start

### 1. Basic Usage

```python
from monitoring import LLMMonitor, MonitoredLLMClient
from openai import OpenAI

# Create monitor
monitor = LLMMonitor()

# Wrap OpenAI client
client = OpenAI(api_key="your-key")
monitored_client = MonitoredLLMClient(client, monitor)

# Use normally, calls are automatically tracked
response = monitored_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    operation="greeting"  # Optional: tag the operation
)

# Export metrics
monitor.export_grafana_json("metrics/llm_metrics.json")
monitor.export_prometheus("metrics/llm_metrics.prom")
monitor.print_summary()
```

### 2. Integration with ResearchAgent

```python
from monitoring import get_global_monitor, MonitoredLLMClient
from openai import OpenAI

# In ResearchAgent.__init__():
self.monitor = get_global_monitor()
raw_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
self.client = MonitoredLLMClient(raw_client, self.monitor, default_operation="research")

# All LLM calls are now tracked automatically
# At the end of execution:
self.monitor.print_summary()
self.monitor.export_grafana_json("ResearchAgent/metrics/llm_metrics.json")
```

### 3. Integration with TransportAgent

```python
from monitoring import get_global_monitor, MonitoredLLMClient

# Similar integration as ResearchAgent
self.monitor = get_global_monitor()
raw_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
self.client = MonitoredLLMClient(raw_client, self.monitor, default_operation="transport")
```

## Metrics Output

### Console Summary

```
================================================================================
LLM METRICS SUMMARY
================================================================================

Overview:
  Total Calls:     25
  Errors:          0
  Success Rate:    100.0%
  Elapsed Time:    45.23s
  Calls/Minute:    33.2

Token Usage:
  Input Tokens:    12,543
  Output Tokens:   3,421
  Total Tokens:    15,964
  Avg/Call:        638

Cost:
  Total Cost:      $0.0523
  Avg/Call:        $0.0021

Latency:
  Average:         1,234ms
  Min:             456ms
  Max:             3,210ms
  P95:             2,567ms

By Operation:
  tag_generation:
    Calls:         10
    Avg Latency:   850ms
    Avg Tokens:    234
    Success Rate:  100.0%

  interest_mapping:
    Calls:         5
    Avg Latency:   1,100ms
    Avg Tokens:    512
    Success Rate:  100.0%
```

### JSON Export (Grafana-compatible)

```json
{
  "timestamp": "2025-10-11T04:30:15.123456",
  "summary": {
    "overview": {
      "total_calls": 25,
      "total_errors": 0,
      "success_rate": 1.0,
      "elapsed_time_seconds": 45.23,
      "calls_per_minute": 33.2
    },
    "tokens": {
      "total_input": 12543,
      "total_output": 3421,
      "total": 15964,
      "avg_per_call": 638.56
    },
    "cost": {
      "total_usd": 0.0523,
      "avg_per_call_usd": 0.0021
    },
    "latency": {
      "avg_ms": 1234.5,
      "min_ms": 456.0,
      "max_ms": 3210.0,
      "p95_ms": 2567.0
    },
    "by_operation": {
      "tag_generation": {
        "count": 10,
        "avg_latency_ms": 850.0,
        "total_tokens": 2340,
        "avg_tokens": 234.0,
        "errors": 0,
        "success_rate": 1.0
      }
    }
  },
  "metrics": [
    {
      "timestamp": "2025-10-11T04:30:10.123456",
      "model": "gpt-4o",
      "operation": "tag_generation",
      "latency_ms": 850.5,
      "input_tokens": 180,
      "output_tokens": 54,
      "total_tokens": 234,
      "cost_usd": 0.00099,
      "success": true,
      "error": null
    }
  ]
}
```

### Prometheus Export

```
# HELP llm_calls_total Total number of LLM API calls
# TYPE llm_calls_total counter
llm_calls_total 25

# HELP llm_errors_total Total number of LLM API errors
# TYPE llm_errors_total counter
llm_errors_total 0

# HELP llm_tokens_input_total Total input tokens
# TYPE llm_tokens_input_total counter
llm_tokens_input_total 12543

# HELP llm_tokens_output_total Total output tokens
# TYPE llm_tokens_output_total counter
llm_tokens_output_total 3421

# HELP llm_cost_usd_total Total cost in USD
# TYPE llm_cost_usd_total counter
llm_cost_usd_total 0.0523

# HELP llm_latency_ms LLM call latency in milliseconds
# TYPE llm_latency_ms histogram
llm_latency_ms{model="gpt-4o",operation="tag_generation"} 850.5
```

## Grafana Setup

### Option 1: JSON API Datasource

1. Install JSON API plugin in Grafana
2. Add datasource pointing to your metrics JSON file
3. Create dashboards using the metrics

### Option 2: Prometheus

1. Configure Prometheus to scrape the `.prom` file
2. Add Prometheus datasource in Grafana
3. Import pre-built dashboard (see `grafana/dashboard.json`)

### Key Metrics to Monitor

- **LLM Call Rate**: `llm_calls_total` - Track API usage over time
- **Token Usage**: `llm_tokens_input_total + llm_tokens_output_total` - Monitor costs
- **Latency**: `llm_latency_ms` - P50, P95, P99 percentiles
- **Error Rate**: `llm_errors_total / llm_calls_total` - API reliability
- **Cost**: `llm_cost_usd_total` - Budget tracking
- **By Operation**: Group all metrics by operation type

## Operation Tags

Use descriptive operation tags to categorize LLM calls:

**ResearchAgent:**
- `interest_mapping` - Mapping user interests to categories
- `tag_generation` - Generating place tags
- `description_enrichment` - Creating descriptions

**TransportAgent:**
- `route_analysis` - Analyzing transport routes
- `carbon_calculation` - Carbon scoring

## Pricing

Default pricing (as of 2024):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |
| gpt-3.5-turbo | $0.50 | $1.50 |

Update prices in `monitoring.py` -> `LLMMonitor.PRICING`

## Best Practices

1. **Always tag operations**: Use descriptive operation names for better analysis
2. **Export regularly**: Export metrics after each agent run
3. **Monitor P95 latency**: Set alerts for high latency
4. **Track cost trends**: Monitor daily/weekly spending
5. **Review error rates**: Investigate any errors immediately

## Troubleshooting

**Issue**: Metrics not exported
- Check write permissions on metrics directory
- Verify directory exists: `mkdir -p metrics/`

**Issue**: Wrong token counts
- Ensure you're using official OpenAI client
- Check model name matches pricing table

**Issue**: High latency
- Check network connectivity
- Consider using faster models (gpt-4o-mini)
- Review prompt complexity

## Examples

See `examples/monitoring_demo.py` for complete integration examples.
