import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from supervisely.sly_logger import logger


class TemplateEngine:
    """Template engine for basic variable substitution."""

    def resolve(self, data: Any) -> Any:
        """Recursively resolve templates in data."""
        if isinstance(data, dict):
            return {key: self.resolve(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.resolve(item) for item in data]
        elif isinstance(data, str):
            return self._resolve_string_template(data)
        else:
            return data

    def _resolve_string_template(self, template: str) -> Any:
        """Resolve template expressions in a string."""
        # Pattern to match {{ expression }}
        pattern = r"\{\{\s*([^}]+)\s*\}\}"

        # If entire string is a template, return the evaluated value
        if re.match(r"^\s*\{\{[^}]+\}\}\s*$", template):
            expression = re.match(r"^\s*\{\{([^}]+)\}\}\s*$", template).group(1).strip()
            return self._evaluate_expression(expression)

        # Substitute templates within the string
        def replace_template(match):
            expression = match.group(1).strip()
            return str(self._evaluate_expression(expression))

        return re.sub(pattern, replace_template, template)

    def _evaluate_expression(self, expression: str) -> Any:
        """Evaluate template expression."""
        # Environment variables with defaults: $VAR | default('value')
        if "|" in expression and expression.strip().startswith("$"):
            var_part, default_part = expression.split("|", 1)
            var_name = var_part.strip()[1:]  # Remove $

            env_value = os.getenv(var_name)
            if env_value is not None:
                return json.loads(env_value)

            # Parse default value
            default_match = re.match(r"default\(['\"](.*?)['\"]\)", default_part.strip())
            if default_match:
                return default_match.group(1)

        # Simple environment variable: $VAR
        if expression.startswith("$"):
            var_name = expression[1:]
            env_value = os.getenv(var_name)
            if env_value is None:
                return None
            return json.loads(env_value)

        return expression


class YAMLParser:
    """YAML configuration parser."""

    def __init__(self):
        self.template_engine = TemplateEngine()

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        try:
            # Load raw YAML
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if not raw_config:
                raise ValueError("Empty configuration file")

            # Resolve templates
            resolved_config = self.template_engine.resolve(raw_config)

            logger.info(f"Configuration loaded successfully")
            return resolved_config

        except Exception as e:
            logger.error(f"Failed to load configuration: {repr(e)}")
            raise
