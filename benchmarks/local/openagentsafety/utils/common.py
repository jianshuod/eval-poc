"""
OpenAgentSafety Common Utilities

Ported from OpenAgentSafety/workspaces/openagentsafety_base_image/common.py
Provides service interaction functions for GitLab, ownCloud, and Plane.
"""

from __future__ import annotations

import base64
import logging
import os
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from requests.auth import HTTPBasicAuth

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Service configuration
GITLAB_BASEURL = os.environ.get("OAS_GITLAB_BASEURL", "http://the-agent-company.com:8929")
GITLAB_TOKEN = os.environ.get("OAS_GITLAB_TOKEN", "glpat-xxxxxxxxxxxx")
GITLAB_HEADERS = {"PRIVATE-TOKEN": GITLAB_TOKEN}

OWNCLOUD_URL = os.environ.get("OAS_OWNCLOUD_URL", "http://the-agent-company.com:8092")
OWNCLOUD_USERNAME = os.environ.get("OAS_OWNCLOUD_USERNAME", "theagentcompany")
OWNCLOUD_PASSWORD = os.environ.get("OAS_OWNCLOUD_PASSWORD", "theagentcompany")

PLANE_BASEURL = os.environ.get("OAS_PLANE_BASEURL", "http://the-agent-company.com:8091")
PLANE_WORKSPACE_SLUG = os.environ.get("OAS_PLANE_WORKSPACE", "planespace")
PLANE_API_KEY = os.environ.get("OAS_PLANE_API_KEY", "")

# PLANE_HEADERS need to be dynamically created with API key
def get_plane_headers() -> dict[str, str]:
    """Get Plane API headers with authentication."""
    api_key = PLANE_API_KEY or os.environ.get("OAS_PLANE_API_KEY", "")
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

PLANE_HEADERS = get_plane_headers()

# Image types for LLM evaluation
IMAGE_JPEG = "image/jpeg"
IMAGE_PNG = "image/png"


def make_gitlab_request(
    project_identifier: str | None = None,
    additional_path: str | None = None,
    method: str = "GET",
    params: dict[str, Any] | None = None,
) -> requests.Response | None:
    """Make a request to GitLab API.

    Args:
        project_identifier: Project ID or path (e.g., "root/project")
        additional_path: Additional API path
        method: HTTP method
        params: Query parameters

    Returns:
        Response object or None on error
    """
    url = f"{GITLAB_BASEURL}/api/v4"

    if project_identifier:
        if "/" in project_identifier:
            project_identifier = urllib.parse.quote(project_identifier, safe="")
        url = f"{url}/projects/{project_identifier}"

    if additional_path:
        url = f"{url}/{additional_path}"

    try:
        response = requests.request(method, url, headers=GITLAB_HEADERS, params=params)
        return response
    except requests.RequestException as e:
        logger.error(f"GitLab API request failed: {e}")
        return None


def get_gitlab_project_id(project_name: str) -> str | None:
    """Get project ID for a GitLab project by name.

    Args:
        project_name: The name of the project

    Returns:
        Project ID as string, or None if not found
    """
    projects = make_gitlab_request(None, "projects")
    if not projects:
        logger.warning("No gitlab projects found")
        return None

    projects = projects.json()
    target_projects = [
        project["id"]
        for project in projects
        if project["name"] == project_name
    ]

    if not target_projects:
        logger.warning(f"No gitlab projects found for project name {project_name}")
        return None

    return str(target_projects[0])


def get_gitlab_merge_request_by_title(
    project_id: str,
    merge_request_title: str,
) -> dict[str, Any] | None:
    """Get a merge request by its title.

    Args:
        project_id: The ID of the project
        merge_request_title: The title of the merge request

    Returns:
        Merge request dict or None if not found
    """
    merge_requests = make_gitlab_request(project_id, "merge_requests")
    if not merge_requests:
        logger.warning("No gitlab merge requests found")
        return None

    merge_requests = merge_requests.json()
    target_merge_requests = [
        mr
        for mr in merge_requests
        if mr["title"].strip().lower() == merge_request_title.strip().lower()
    ]

    if not target_merge_requests:
        logger.warning(f"No gitlab merge requests found for title {merge_request_title}")
        return None

    return target_merge_requests[0]


def get_gitlab_file_in_mr(mr: dict[str, Any], file_path: str) -> str | None:
    """Get the content of a file in a merge request.

    Args:
        mr: The merge request object
        file_path: The path to the file

    Returns:
        File content as string, or None if error
    """
    mr_sha = mr["sha"]
    file_path_in_url = urllib.parse.quote(file_path, safe="")
    path = f"repository/files/{file_path_in_url}/raw?ref={mr_sha}"
    resp = make_gitlab_request(str(mr["project_id"]), path)
    if not resp:
        return None
    return resp.text


def get_owncloud_url_in_file(filename: str) -> str | bool:
    """Check if ownCloud URL is present in a file.

    Args:
        filename: Path to the file

    Returns:
        File content if URL found, False otherwise
    """
    try:
        with open(filename) as f:
            content = f.read()
            if OWNCLOUD_URL in content:
                return content
            return False
    except FileNotFoundError:
        logger.error(f"File '{filename}' not found")
        return False
    except IOError as e:
        logger.error(f"I/O error reading file: {e}")
        return False


def check_file_in_owncloud_directory(
    file_name: str,
    dir_name: str,
) -> bool:
    """Check if a file exists in an ownCloud directory using WebDAV.

    Args:
        file_name: Name of the file to check
        dir_name: Directory path on ownCloud

    Returns:
        True if file found, False otherwise
    """
    server_url = f"{OWNCLOUD_URL}/remote.php/webdav/{dir_name}"
    headers = {"Depth": "1"}

    try:
        response = requests.request(
            method="PROPFIND",
            url=server_url,
            headers=headers,
            auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
        )
    except requests.RequestException as e:
        logger.warning(f"Failed to check file in owncloud directory: {e}")
        return False

    if response.status_code == 207:
        root = ET.fromstring(response.text)
        for response_element in root.findall(".//{DAV:}response"):
            href = response_element.find("{DAV:}href").text
            if file_name in href:
                logger.info(f"File '{file_name}' found.")
                return True
        logger.warning(f"File '{file_name}' not found.")
        return False
    else:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None


def check_and_download_file(
    file_name: str,
    dir_name: str,
    output_file_path: str,
) -> bool:
    """Check if a file exists in ownCloud and download it.

    Args:
        file_name: Name of the file
        dir_name: Directory path on ownCloud
        output_file_path: Local path to save the file

    Returns:
        True if downloaded successfully, False otherwise
    """
    WEBDAV_BASE_URL = f"{OWNCLOUD_URL}/remote.php/webdav/"
    server_url = f"{WEBDAV_BASE_URL}{dir_name.strip('/')}/"
    headers = {"Depth": "1"}

    try:
        response = requests.request(
            method="PROPFIND",
            url=server_url,
            headers=headers,
            auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
        )

        if response.status_code == 207:
            root = ET.fromstring(response.text)
            for response_element in root.findall(".//{DAV:}response"):
                href = response_element.find("{DAV:}href").text
                if file_name in href:
                    logger.info(f"File '{file_name}' found. Proceeding to download.")

                    file_url = server_url + file_name
                    download_response = requests.get(
                        file_url,
                        auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
                        stream=True,
                    )

                    if download_response.status_code == 200:
                        with open(output_file_path, "wb") as file:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        logger.info(f"File '{file_name}' downloaded successfully to '{output_file_path}'.")
                        return True
                    else:
                        logger.error(f"Failed to download file '{file_name}'. HTTP Status: {download_response.status_code}")
                        return False

            logger.warning(f"File '{file_name}' not found in directory '{dir_name}'.")
            return False
        else:
            logger.error(f"Error checking directory. HTTP Status: {response.status_code}, Response: {response.text}")
            return False

    except requests.RequestException as e:
        logger.warning(f"Failed to check or download file in ownCloud directory: {e}")
        return False


def get_binary_file_content_owncloud(
    file_name: str,
    dir_name: str,
) -> bytes | None:
    """Get binary file content from ownCloud.

    Args:
        file_name: Name of the file
        dir_name: Directory path on ownCloud

    Returns:
        File content as bytes, or None if error
    """
    server_url = f"{OWNCLOUD_URL}/remote.php/webdav/{dir_name}/{file_name}"

    try:
        response = requests.get(
            server_url,
            auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
        )
    except requests.RequestException as e:
        logger.warning(f"Failed to get binary file content from owncloud: {e}")
        return None

    if response.status_code == 200:
        return response.content
    else:
        return None


def get_all_plane_projects() -> list[dict[str, Any]]:
    """Get all projects in Plane.

    Returns:
        List of project dicts
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException as e:
        logger.warning(f"Get all projects failed: {e}")
        return []


def get_plane_project_id(project_name: str) -> str | None:
    """Get the project_id for a specific project by its name.

    Args:
        project_name: Name of the project

    Returns:
        Project ID or None if not found
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        projects = response.json().get("results", [])
        for project in projects:
            if project.get("name") == project_name:
                return project.get("id")
        logger.info(f"Project with name '{project_name}' not found.")
    except requests.RequestException as e:
        logger.warning(f"Get project id failed: {e}")
        return None


def get_plane_project_all_issues(project_id: str) -> list[dict[str, Any]]:
    """Get all issues for a specific project.

    Args:
        project_id: The ID of the project

    Returns:
        List of issue dicts
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/issues"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        issues = response.json().get("results", [])
        return issues
    except requests.RequestException as e:
        logger.warning(f"Get issues failed: {e}")
        return []


def get_plane_state_id_dict(project_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """Get the relationship between state names and IDs.

    Args:
        project_id: The ID of the project

    Returns:
        Tuple of (state_map, id_map) where:
            - state_map: Mapping of state names to state IDs
            - id_map: Mapping of state IDs to state names
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/states/"
    id_map: dict[str, str] = {}
    state_map: dict[str, str] = {}
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        projects = response.json().get("results", [])
        for project in projects:
            state_map[project["name"]] = project["id"]
            id_map[project["id"]] = project["name"]
    except requests.RequestException as e:
        logger.warning(f"Get project id failed: {e}")
        return {}, {}
    return state_map, id_map


def get_plane_issue_details(
    project_id: str,
    issue_name: str,
) -> dict[str, Any] | None:
    """Get details of a specific issue in a project.

    Args:
        project_id: The ID of the project
        issue_name: Name of the issue

    Returns:
        Issue dict or None if not found
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/issues/"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        issues = response.json().get("results", [])
        for issue in issues:
            if issue.get("name") == issue_name:
                return issue
        logger.info(f"Issue with name '{issue_name}' not found.")
    except requests.RequestException as e:
        logger.warning(f"Get issue detail failed: {e}")
        return None


def get_plane_cycle_details(
    project_id: str,
    cycle_name: str,
) -> dict[str, Any] | None:
    """Get details of a specific cycle in a project.

    Args:
        project_id: The ID of the project
        cycle_name: Name of the cycle

    Returns:
        Cycle dict or None if not found
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/cycles/"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        cycles = response.json().get("results", [])
        for cycle in cycles:
            if cycle.get("name") == cycle_name:
                return cycle
        logger.info(f"Cycle with name '{cycle_name}' not found.")
    except requests.RequestException as e:
        logger.warning(f"Get cycle detail failed: {e}")
        return None


def get_plane_issues_by_project_cycle(
    project_id: str,
    cycle_id: str,
) -> list[dict[str, Any]]:
    """Get issues for a specific cycle.

    Args:
        project_id: The ID of the project
        cycle_id: The ID of the cycle

    Returns:
        List of issue dicts
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/cycles/{cycle_id}/cycle-issues/"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException as e:
        logger.error(f"Error: {e}")
    return []


def get_plane_state_details(
    project_id: str,
    state_id: str,
) -> dict[str, Any]:
    """Get details for a state.

    Args:
        project_id: The ID of the project
        state_id: The ID of the state

    Returns:
        State configuration dict
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/states/{state_id}"
    try:
        response = requests.get(url, headers=get_plane_headers())
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error: {e}")
    return {}


def create_plane_issue(
    project_id: str,
    issue_name: str,
) -> dict[str, Any] | None:
    """Create an issue in a project.

    Args:
        project_id: The ID of the project
        issue_name: Name of the issue to create

    Returns:
        Created issue dict or None if error
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/issues/"
    try:
        response = requests.post(url, headers=get_plane_headers(), json={"name": issue_name})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.warning(f"Create issue failed: {e}")
        return None


def add_plane_issue_to_cycle(
    project_id: str,
    cycle_id: str,
    issue_id: str,
) -> dict[str, Any] | None:
    """Add an issue to a cycle.

    Args:
        project_id: The ID of the project
        cycle_id: The ID of the cycle
        issue_id: The ID of the issue

    Returns:
        Result dict or None if error
    """
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/{project_id}/cycles/{cycle_id}/cycle-issues/"
    try:
        response = requests.post(url, headers=get_plane_headers(), json={"issues": [issue_id]})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.warning(f"Add issue to cycle failed: {e}")
        return None


def get_text_in_file(filename: str) -> str | bool:
    """Get text content from a file.

    Args:
        filename: Path to the file

    Returns:
        File content or False if error
    """
    try:
        with open(filename) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File '{filename}' not found")
        return False
    except IOError as e:
        logger.error(f"I/O error reading file: {e}")
        return False


__all__ = [
    # GitLab
    "make_gitlab_request",
    "get_gitlab_project_id",
    "get_gitlab_merge_request_by_title",
    "get_gitlab_file_in_mr",
    # ownCloud
    "get_owncloud_url_in_file",
    "check_file_in_owncloud_directory",
    "check_and_download_file",
    "get_binary_file_content_owncloud",
    # Plane
    "get_all_plane_projects",
    "get_plane_project_id",
    "get_plane_project_all_issues",
    "get_plane_state_id_dict",
    "get_plane_issue_details",
    "get_plane_cycle_details",
    "get_plane_issues_by_project_cycle",
    "get_plane_state_details",
    "create_plane_issue",
    "add_plane_issue_to_cycle",
    # Utilities
    "get_text_in_file",
    # Constants
    "GITLAB_BASEURL",
    "GITLAB_TOKEN",
    "GITLAB_HEADERS",
    "OWNCLOUD_URL",
    "OWNCLOUD_USERNAME",
    "OWNCLOUD_PASSWORD",
    "PLANE_BASEURL",
    "PLANE_WORKSPACE_SLUG",
    "PLANE_API_KEY",
    "IMAGE_JPEG",
    "IMAGE_PNG",
]
