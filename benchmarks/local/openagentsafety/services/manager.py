"""
OpenAgentSafety Docker Service Manager

This module manages the lifecycle of Docker services required for OAS evaluation:
- GitLab (code repository)
- ownCloud (file sharing)
- Plane (project management)
- RocketChat (messaging)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ServiceName(Enum):
    """Available OAS services."""

    GITLAB = "gitlab"
    OWN_CLOUD = "owncloud"
    PLANE = "plane"
    ROCKET_CHAT = "rocketchat"


# Service startup times (in seconds) - health check timeout
SERVICE_STARTUP_TIMES = {
    ServiceName.GITLAB: 180,  # GitLab takes the longest
    ServiceName.OWN_CLOUD: 60,
    ServiceName.PLANE: 90,
    ServiceName.ROCKET_CHAT: 90,
}

# Service port mappings
SERVICE_PORTS = {
    ServiceName.GITLAB: 8929,
    ServiceName.OWN_CLOUD: 8092,
    ServiceName.PLANE: 8091,
    ServiceName.ROCKET_CHAT: 3000,
}


@dataclass
class ServiceStatus:
    """Status of a Docker service."""

    name: ServiceName
    running: bool
    healthy: bool
    port: int | None = None
    error: str | None = None


logger = logging.getLogger(__name__)


class OASServiceManager:
    """Manager for OAS Docker services.

    This class handles starting, stopping, and checking the health of
    Docker services required for OpenAgentSafety evaluation.
    """

    def __init__(
        self,
        compose_file: str | Path | None = None,
        project_name: str = "oas-eval",
    ):
        """Initialize the service manager.

        Args:
            compose_file: Path to docker-compose.yaml file
            project_name: Docker Compose project name
        """
        if compose_file is None:
            # Default to services/compose.yaml in the same directory
            compose_file = (
                Path(__file__).parent / "compose.yaml"
            )

        self.compose_file = Path(compose_file)
        self.project_name = project_name
        self._started_services: set[ServiceName] = set()

    def _run_compose_command(
        self,
        command: list[str],
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a docker-compose command.

        Args:
            command: Command arguments (without 'docker compose')
            capture: Whether to capture stdout/stderr

        Returns:
            CompletedProcess with the result
        """
        full_cmd = ["docker", "compose", "-f", str(self.compose_file)]
        full_cmd.extend(["-p", self.project_name])
        full_cmd.extend(command)

        logger.debug(f"Running command: {' '.join(full_cmd)}")

        return subprocess.run(
            full_cmd,
            capture_output=capture,
            text=not capture,
            check=False,
        )

    def start_services(
        self,
        services: list[ServiceName] | None = None,
        wait_for_healthy: bool = True,
        timeout: int = 300,
    ) -> dict[ServiceName, ServiceStatus]:
        """Start Docker services.

        Args:
            services: List of services to start (None = all)
            wait_for_healthy: Wait for services to be healthy
            timeout: Timeout for health checks in seconds

        Returns:
            Dict of service name -> ServiceStatus
        """
        if services is None:
            services = list(ServiceName)

        # Build service names for docker compose
        service_names = [s.value for s in services]

        logger.info(f"Starting services: {', '.join(service_names)}")

        # Start services
        result = self._run_compose_command(
            ["up", "-d"] + service_names,
            capture=False,
        )

        if result.returncode != 0:
            logger.error(f"Failed to start services: {result.stderr}")
            raise RuntimeError(f"docker compose up failed: {result.stderr}")

        # Track started services
        self._started_services.update(services)

        # Wait for health if requested
        if wait_for_healthy:
            return self.wait_for_services_healthy(services, timeout)

        # Return initial status
        return self.get_service_status(services)

    def stop_services(
        self,
        services: list[ServiceName] | None = None,
        remove_volumes: bool = False,
    ) -> dict[ServiceName, ServiceStatus]:
        """Stop Docker services.

        Args:
            services: List of services to stop (None = all started)
            remove_volumes: Whether to remove volumes

        Returns:
            Dict of service name -> ServiceStatus (all should be not running)
        """
        if services is None:
            services = list(self._started_services)

        if not services:
            return {}

        service_names = [s.value for s in services]
        logger.info(f"Stopping services: {', '.join(service_names)}")

        # Stop services
        cmd = ["stop"] + service_names
        result = self._run_compose_command(cmd, capture=False)

        if result.returncode != 0:
            logger.warning(f"Failed to stop services: {result.stderr}")

        # Optionally remove volumes
        if remove_volumes:
            logger.info("Removing volumes...")
            self._run_compose_command(
                ["down", "-v"],
                capture=False,
            )

        # Update tracking
        for service in services:
            self._started_services.discard(service)

        return {s: ServiceStatus(s, False, False) for s in services}

    def get_service_status(
        self,
        services: list[ServiceName] | None = None,
    ) -> dict[ServiceName, ServiceStatus]:
        """Get current status of services.

        Args:
            services: List of services to check (None = all)

        Returns:
            Dict of service name -> ServiceStatus
        """
        if services is None:
            services = list(ServiceName)

        # Run docker compose ps
        result = self._run_compose_command(["ps", "--format", "json"])

        status_map: dict[ServiceName, ServiceStatus] = {}

        if result.returncode == 0:
            import json

            try:
                running_containers = json.loads(result.stdout)
                running_names = {
                    c["Name"].replace(f"{self.project_name}-", "").replace("-1", "")
                    for c in running_containers
                    if c.get("State") == "running"
                }
            except json.JSONDecodeError:
                running_names = set()
        else:
            running_names = set()

        for service in services:
            name = service.value
            running = name in running_names
            port = SERVICE_PORTS.get(service)

            status_map[service] = ServiceStatus(
                name=service,
                running=running,
                healthy=False,  # Will be updated by health check
                port=port,
            )

        return status_map

    def wait_for_services_healthy(
        self,
        services: list[ServiceName],
        timeout: int = 300,
        check_interval: int = 5,
    ) -> dict[ServiceName, ServiceStatus]:
        """Wait for services to become healthy.

        Args:
            services: List of services to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between health checks

        Returns:
            Dict of service name -> ServiceStatus
        """
        from .health_check import check_services_healthy

        start_time = time.time()

        while time.time() - start_time < timeout:
            health_status = check_services_healthy(services)

            all_healthy = all(s.healthy for s in health_status.values())

            if all_healthy:
                logger.info("All services are healthy")
                return health_status

            # Log waiting status
            waiting = [s.name.value for s, status in health_status.items() if not status.healthy]
            logger.debug(f"Waiting for services: {', '.join(waiting)}")

            time.sleep(check_interval)

        # Timeout reached
        logger.warning(f"Timeout waiting for services after {timeout}s")
        return health_status

    def cleanup(self) -> None:
        """Stop all started services and clean up."""
        if self._started_services:
            logger.info("Cleaning up services...")
            self.stop_services(remove_volumes=False)


# Convenience functions for standalone use
def start_services(
    services: list[ServiceName] | None = None,
    compose_file: str | Path | None = None,
    timeout: int = 300,
) -> OASServiceManager:
    """Start OAS services and return a manager instance.

    Args:
        services: List of services to start (None = all)
        compose_file: Path to docker-compose.yaml
        timeout: Health check timeout

    Returns:
        OASServiceManager instance with started services
    """
    manager = OASServiceManager(compose_file=compose_file)
    manager.start_services(services=services, timeout=timeout)
    return manager


def stop_services(
    manager: OASServiceManager | None = None,
    services: list[ServiceName] | None = None,
    compose_file: str | Path | None = None,
) -> None:
    """Stop OAS services.

    Args:
        manager: OASServiceManager instance (if None, creates new one)
        services: List of services to stop (None = all)
        compose_file: Path to docker-compose.yaml (if manager not provided)
    """
    if manager is None:
        manager = OASServiceManager(compose_file=compose_file)

    manager.stop_services(services=services)


__all__ = [
    "ServiceName",
    "ServiceStatus",
    "OASServiceManager",
    "SERVICE_PORTS",
    "SERVICE_STARTUP_TIMES",
    "start_services",
    "stop_services",
]
