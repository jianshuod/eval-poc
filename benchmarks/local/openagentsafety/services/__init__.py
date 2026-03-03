"""
OpenAgentSafety Service Management

This package provides Docker service management for OAS evaluation.
"""

from .health_check import (
    DEFAULT_CREDENTIALS,
    HEALTH_CHECKS,
    SERVICE_ENDPOINTS,
    check_gitlab_health,
    check_owncloud_health,
    check_plane_health,
    check_rocketchat_health,
    check_service_health,
    check_services_healthy,
    wait_for_service,
    wait_for_services,
)
from .manager import (
    OASServiceManager,
    SERVICE_PORTS,
    SERVICE_STARTUP_TIMES,
    ServiceName,
    ServiceStatus,
    start_services,
    stop_services,
)

__all__ = [
    # Manager
    "OASServiceManager",
    "ServiceName",
    "ServiceStatus",
    "start_services",
    "stop_services",
    "SERVICE_PORTS",
    "SERVICE_STARTUP_TIMES",
    # Health checks
    "check_gitlab_health",
    "check_owncloud_health",
    "check_plane_health",
    "check_rocketchat_health",
    "check_service_health",
    "check_services_healthy",
    "wait_for_service",
    "wait_for_services",
    "HEALTH_CHECKS",
    "SERVICE_ENDPOINTS",
    "DEFAULT_CREDENTIALS",
]
