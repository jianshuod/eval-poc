"""
OpenAgentSafety Utilities

This package contains ported utilities from OpenAgentSafety for use in eval-poc.
"""

from . import evaluators
from .common import (
    IMAGE_JPEG,
    IMAGE_PNG,
    GITLAB_BASEURL,
    GITLAB_TOKEN,
    GITLAB_HEADERS,
    OWNCLOUD_URL,
    OWNCLOUD_USERNAME,
    OWNCLOUD_PASSWORD,
    PLANE_BASEURL,
    PLANE_WORKSPACE_SLUG,
    PLANE_API_KEY,
    # GitLab functions
    make_gitlab_request,
    get_gitlab_project_id,
    get_gitlab_merge_request_by_title,
    get_gitlab_file_in_mr,
    # ownCloud functions
    get_owncloud_url_in_file,
    check_file_in_owncloud_directory,
    check_and_download_file,
    get_binary_file_content_owncloud,
    # Plane functions
    get_all_plane_projects,
    get_plane_project_id,
    get_plane_project_all_issues,
    get_plane_state_id_dict,
    get_plane_issue_details,
    get_plane_cycle_details,
    get_plane_issues_by_project_cycle,
    get_plane_state_details,
    create_plane_issue,
    add_plane_issue_to_cycle,
    # Utilities
    get_text_in_file,
)
from .scoring import (
    Checkpoint,
    Result,
    bonus_for_completing_final,
    bonus_for_completing_any,
    bonus_for_completing_any_of_given_checkpoints,
)

__all__ = [
    # Evaluators
    "evaluators",
    # Scoring
    "Checkpoint",
    "Result",
    "bonus_for_completing_final",
    "bonus_for_completing_any",
    "bonus_for_completing_any_of_given_checkpoints",
    # Common utilities
    "IMAGE_JPEG",
    "IMAGE_PNG",
    "GITLAB_BASEURL",
    "GITLAB_TOKEN",
    "GITLAB_HEADERS",
    "OWNCLOUD_URL",
    "OWNCLOUD_USERNAME",
    "OWNCLOUD_PASSWORD",
    "PLANE_BASEURL",
    "PLANE_WORKSPACE_SLUG",
    "PLANE_API_KEY",
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
]
