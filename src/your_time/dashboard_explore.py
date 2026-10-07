"""Offline context exploration and calendar helpers for the dashboard.

Context presentation can be more specific than compact task rollups, but must
reconcile to the same identified time. No inferred label is treated as an outcome.
"""

from dashboard_assets import read_asset

CONTEXT_JS = read_asset("contexts.js")

INSPECTOR_JS = read_asset("inspector.js")
