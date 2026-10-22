"""Assignment entry points for weak normalization and Hellings queries."""

from project.hellings import hellings_based_cfpq
from project.weak_normal_form import cfg_to_weak_normal_form

__all__ = ["cfg_to_weak_normal_form", "hellings_based_cfpq"]
