"""
OpenAgentSafety Service Health Checks

This module provides health check functions for OAS Docker services.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import requests
from requests.exceptions import RequestException

if TYPE_CHECKING:
    pass

from .manager import ServiceName, ServiceStatus, SERVICE_PORTS

logger = logging.getLogger(__name__)

# Default credentials for services
DEFAULT_CREDENTIALS = {
    ServiceName.GITLAB: {
        "username": "root",
        "password": "12345678",
    },
    ServiceName.OWN_CLOUD: {
        "username": "theagentcompany",
        "password": "theagentcompany",
    },
    ServiceName.PLANE: {
        "email": "admin@example.com",
        "password": "theagentcompany",
    },
    ServiceName.ROCKET_CHAT: {
        "username": "theagentcompany",
        "password": "theagentcompany",
    },
}

# Service health check endpoints
SERVICE_ENDPOINTS = {
    ServiceName.GITLAB: "http://localhost:{port}/api/v4/",
    ServiceName.OWN_CLOUD: "http://localhost:{port}/status.php",
    ServiceName.PLANE: "http://localhost:{port}/api/v1/",
    ServiceName.ROCKET_CHAT: "http://localhost:{port}/api/info",
}


def check_gitlab_health(timeout: int = 10) -> bool:
    """Check if GitLab is healthy.

    Args:
        timeout: Request timeout in seconds

    Returns:
        True if GitLab is healthy
    """
    port = SERVICE_PORTS.get(ServiceName.GITLAB, 8929)
    url = SERVICE_ENDPOINTS[ServiceName.GITLAB].format(port=port)

    try:
        response = requests.get(url, timeout=timeout)
        # GitLab API returns 401 when no auth, but that's healthy
        return response.status_code in {200, 401}
    except RequestException as e:
        logger.debug(f"GitLab health check failed: {e}")
        return False


def check_owncloud_health(timeout: int = 10) -> bool:
    """Check if ownCloud is healthy.

    Args:
        timeout: Request timeout in seconds

    Returns:
        True if ownCloud is healthy
    """
    port = SERVICE_PORTS.get(ServiceName.OWN_CLOUD, 8092)
    url = SERVICE_ENDPOINTS[ServiceName.OWN_CLOUD].format(port=port)

    try:
        response = requests.get(url, timeout=timeout)
        # status.php returns {"installed":"true","maintenance":"false"}
        return response.status_code == 200
    except RequestException as e:
        logger.debug(f"ownCloud health check failed: {e}")
        return False


def check_plane_health(timeout: int = 10) -> bool:
    """Check if Plane is healthy.

    Args:
        timeout: Request timeout in seconds

    Returns:
        True if Plane is healthy
    """
    port = SERVICE_PORTS.get(ServiceName.PLANE, 8091)
    url = SERVICE_ENDPOINTS[ServiceName.PLANE].format(port=port)

    try:
        response = requests.get(url, timeout=timeout)
        # Plane API returns 401 for unauthenticated requests
        return response.status_code in {200, 401}
    except RequestException as e:
        logger.debug(f"Plane health check failed: {e}")
        return False


def check_rocketchat_health(timeout: int = 10) -> bool:
    """Check if RocketChat is healthy.

    Args:
        timeout: Request timeout in seconds

    Returns:
        True if RocketChat is healthy
    """
    port = SERVICE_PORTS.get(ServiceName.ROCKET_CHAT, 3000)
    url = SERVICE_ENDPOINTS[ServiceName.ROCKET_CHAT].format(port=port)

    try:
        response = requests.get(url, timeout=timeout)
        # RocketChat /api/info returns JSON with version
        return response.status_code == 200 and "version" in response.text.lower()
    except RequestException as e:
        logger.debug(f"RocketChat health check failed: {e}")
        return False


# Health check function mapping
HEALTH_CHECKS = {
    ServiceName.GITLAB: check_gitlab_health,
    ServiceName.OWN_CLOUD: check_owncloud_health,
    ServiceName.PLANE: check_plane_health,
    ServiceName.ROCKET_CHAT: check_rocketchat_health,
}


def check_service_health(service: ServiceName, timeout: int = 10) -> ServiceStatus:
    """Check health of a single service.

    Args:
        service: Service to check
        timeout: Request timeout

    Returns:
        ServiceStatus with health information
    """
    check_fn = HEALTH_CHECKS.get(service)

    if check_fn is None:
        return ServiceStatus(
            name=service,
            running=False,
            healthy=False,
            error=f"No health check for {service.value}",
        )

    try:
        healthy = check_fn(timeout=timeout)
        port = SERVICE_PORTS.get(service)

        return ServiceStatus(
            name=service,
            running=True,
            healthy=healthy,
            port=port,
        )
    except Exception as e:
        return ServiceStatus(
            name=service,
            running=True,
            healthy=False,
            error=str(e),
        )


def check_services_healthy(
    services: list[ServiceName],
    timeout: int = 10,
) -> dict[ServiceName, ServiceStatus]:
    """Check health of multiple services.

    Args:
        services: List of services to check
        timeout: Per-request timeout

    Returns:
        Dict of service name -> ServiceStatus
    """
    results = {}

    for service in services:
        results[service] = check_service_health(service, timeout=timeout)

    return results


def wait_for_service(
    service: ServiceName,
    timeout: int = 180,
    check_interval: int = 5,
) -> bool:
    """Wait for a single service to become healthy.

    Args:
        service: Service to wait for
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks

    Returns:
        True if service became healthy, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = check_service_health(service)

        if status.healthy:
            logger.info(f"{service.value} is healthy")
            return True

        logger.debug(f"Waiting for {service.value}...")
        time.sleep(check_interval)

    logger.warning(f"Timeout waiting for {service.value}")
    return False


def wait_for_services(
    services: list[ServiceName],
    timeout: int = 300,
    check_interval: int = 5,
) -> dict[ServiceName, bool]:
    """Wait for multiple services to become healthy.

    Args:
        services: List of services to wait for
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks

    Returns:
        Dict of service name -> healthy status
    """
    results = {}

    for service in services:
        results[service] = wait_for_service(
            service,
            timeout=timeout,
            check_interval=check_interval,
        )

    return results


__all__ = [
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
