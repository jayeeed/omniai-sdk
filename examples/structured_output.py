"""OmniAI SDK Structured Output with Pydantic Example."""

from typing import List
from pydantic import BaseModel, Field
from omniai import OmniaiClient


class ServiceHealthAnalysis(BaseModel):
    service_name: str = Field(description="Name of the service analyzed")
    status: str = Field(description="Overall status (HEALTHY, DEGRADED, DOWN)")
    key_metrics: List[str] = Field(description="List of key performance indicators")
    recommendations: List[str] = Field(description="Actionable recommendations")


client = OmniaiClient()

result: ServiceHealthAnalysis = client.chat(
    model="gemini:gemini-2.5-flash",
    prompt="Analyze an event-driven payments service experiencing 99.8% uptime with occasional 500ms latency spikes.",
    response_model=ServiceHealthAnalysis,
)

print(f"Service: {result.service_name}")
print(f"Status: {result.status}")
print(f"Metrics: {result.key_metrics}")
print(f"Recommendations: {result.recommendations}")
