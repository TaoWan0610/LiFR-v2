"""Public entries for semantic EGCM and event-encoder distillation."""
from .distillation import RGBEventDistiller
from .completion import GateRefine4Stage

__all__ = ['RGBEventDistiller', 'GateRefine4Stage']
