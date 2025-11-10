"""
Feature flags configuration (deprecated - kept for compatibility).

All modular features are now enabled by default.
"""


class FeatureFlags:
    """Legacy feature flags class for compatibility."""

    def __init__(self):
        """Initialize feature flags with all features enabled by default."""
        # All features enabled by default
        self._flags = {
            "use_database_manager": True,
            "use_sql_generation_service": True,
            "use_error_handling_service": True,
        }

    def get_flag(self, flag_name: str) -> bool:
        """All flags return True."""
        return True

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled (compatibility method)."""
        return True

    def get_all_flags(self) -> dict:
        """Return all flags as True."""
        return self._flags

    def set_rollout_stage(self, stage: str):
        """No-op for compatibility."""
        pass


# Global instance
feature_flags = FeatureFlags()
