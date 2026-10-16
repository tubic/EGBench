"""Exceptions raised by the eDNN-EG implementation."""
class EDNNEGError(RuntimeError):
    pass


class ContractViolation(EDNNEGError):
    pass


class MissingRequiredDependency(EDNNEGError):
    pass
