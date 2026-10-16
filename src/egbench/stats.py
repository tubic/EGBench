"""Stratified bootstrap intervals, paired differences, Holm and Benjamini-Hochberg corrections and Fisher's exact test."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import NamedTuple

import numpy as np

from egbench.contracts import SEED, TargetKind
from egbench.metrics._input import UNDEFINED, check_continuous_pair, check_pair

RESAMPLE_COUNT = 4000

QUARTILE_COUNT = 4

CONFIDENCE_LEVEL = 0.95

UNDEFINED_TOLERANCE = 0.01

_Z = NormalDist().inv_cdf(1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0)

_INTEGER_KINDS = "iu"


class Interval(NamedTuple):

    point: float
    half_width: float
    undefined_fraction: float


class Comparison(NamedTuple):

    difference: float
    half_width: float
    p_value: float
    undefined_fraction: float


class _Fold(NamedTuple):

    targets: np.ndarray
    columns: tuple[np.ndarray, ...]
    strata: tuple[int, ...]


def _fold_identifier(fold: object, count: int) -> np.ndarray:
    if fold is None:
        return np.zeros(count, dtype="int64")
    identifier = np.asarray(fold)
    if identifier.ndim != 1:
        raise ValueError(
            f"fold must be a one dimensional array, its shape is {identifier.shape}"
        )
    if identifier.dtype.kind not in _INTEGER_KINDS:
        raise ValueError(
            f"fold must hold integers, its dtype is {identifier.dtype}; the fold "
            f"table identifies a fold by an integer and a gene belongs to one "
            f"fold or to no fold at all"
        )
    if identifier.size != count:
        raise ValueError(
            f"fold must be as long as y_true, y_true holds {count} values and "
            f"fold holds {identifier.size}"
        )
    return identifier


def _strata(
    targets: np.ndarray,
    positions: np.ndarray,
    columns: Sequence[np.ndarray],
    target_kind: TargetKind,
) -> list[np.ndarray]:
    in_fold = targets[positions]
    if target_kind is TargetKind.BINARY:
        return [
            positions[np.flatnonzero(in_fold)],
            positions[np.flatnonzero(~in_fold)],
        ]
    ordered = positions[
        np.lexsort(
            (*tuple(column[positions] for column in reversed(columns)), in_fold)
        )
    ]
    edge = [
        ordered.size * index // QUARTILE_COUNT for index in range(QUARTILE_COUNT + 1)
    ]
    return [ordered[start:stop] for start, stop in zip(edge, edge[1:])]


def _folds(
    targets: np.ndarray,
    columns: Sequence[np.ndarray],
    fold: object,
    target_kind: TargetKind,
) -> tuple[_Fold, ...]:
    identifier = _fold_identifier(fold, targets.size)
    built = []
    for value in np.unique(identifier):
        positions = np.flatnonzero(identifier == value)
        groups = _strata(targets, positions, columns, target_kind)
        laid_out = np.concatenate(
            [
                stratum[
                    np.lexsort(tuple(column[stratum] for column in reversed(columns)))
                ]
                for stratum in groups
            ]
        )
        built.append(
            _Fold(
                targets=targets[laid_out].astype("float64", copy=False),
                columns=tuple(column[laid_out] for column in columns),
                strata=tuple(int(stratum.size) for stratum in groups),
            )
        )
    return tuple(built)


def _value(
    metric: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    columns: Sequence[np.ndarray],
) -> float:
    total = 0.0
    for index, scores in enumerate(columns):
        value = metric(labels, scores)
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"metric must return a number, it returned {value!r}"
            ) from error
        if math.isnan(value):
            return UNDEFINED
        if math.isinf(value):
            raise ValueError(
                f"metric returned an infinite value, {value!r}; a metric with no "
                f"value returns NaN and one that returns an infinity is wrong"
            )
        total = value if index == 0 else total - value
    return total


def _statistic(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        if math.isnan(value):
            return UNDEFINED
        total += value
    return total / len(values)


def _draw(generator: np.random.Generator, strata: Sequence[int]) -> np.ndarray:
    offset = 0
    parts = []
    for size in strata:
        parts.append(offset + generator.integers(0, size, size))
        offset += size
    return np.concatenate(parts)


def _resample(
    metrics: Sequence[Callable[[np.ndarray, np.ndarray], float]],
    folds: Sequence[_Fold],
) -> np.ndarray:
    generator = np.random.default_rng(SEED)
    replicates = np.empty(RESAMPLE_COUNT, dtype="float64")
    for index in range(RESAMPLE_COUNT):
        total = 0.0
        for metric, fold in zip(metrics, folds):
            take = _draw(generator, fold.strata)
            value = _value(
                metric, fold.targets[take], [column[take] for column in fold.columns]
            )
            if math.isnan(value):
                total = UNDEFINED
                break
            total += value
        replicates[index] = total / len(folds)
    return replicates


def _resample_many(
    metric_sets: dict[str, Sequence[Callable[[np.ndarray, np.ndarray], float]]],
    folds: Sequence[_Fold],
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(SEED)
    names = list(metric_sets)
    replicates = {
        name: np.empty(RESAMPLE_COUNT, dtype="float64") for name in names
    }
    for index in range(RESAMPLE_COUNT):
        totals = dict.fromkeys(names, 0.0)
        for position, fold in enumerate(folds):
            take = _draw(generator, fold.strata)
            targets = fold.targets[take]
            columns = [column[take] for column in fold.columns]
            for name in names:
                if totals[name] != totals[name]:
                    continue
                value = _value(metric_sets[name][position], targets, columns)
                totals[name] = UNDEFINED if math.isnan(value) else totals[name] + value
        for name in names:
            total = totals[name]
            replicates[name][index] = (
                UNDEFINED if total != total else total / len(folds)
            )
    return replicates


class _Bootstrap(NamedTuple):
    point: float
    replicates: np.ndarray
    undefined_fraction: float


def _bootstrap(
    metrics: Sequence[Callable[[np.ndarray, np.ndarray], float]],
    folds: Sequence[_Fold],
) -> _Bootstrap:
    fold_values = [
        _value(metric, fold.targets, fold.columns)
        for metric, fold in zip(metrics, folds)
    ]
    point = _statistic(fold_values)
    if math.isnan(point):
        return _Bootstrap(UNDEFINED, np.empty(0), UNDEFINED)
    replicates = _resample(metrics, folds)
    defined = ~np.isnan(replicates)
    undefined_fraction = 1.0 - float(defined.mean())
    if undefined_fraction > UNDEFINED_TOLERANCE:
        return _Bootstrap(point, np.empty(0), undefined_fraction)
    return _Bootstrap(point, replicates[defined], undefined_fraction)


def _standard_error(replicates: np.ndarray) -> float:
    return float(np.std(replicates, ddof=1))


def _half_width(bootstrap: _Bootstrap) -> float:
    if math.isnan(bootstrap.point) or bootstrap.replicates.size < 2:
        return UNDEFINED
    return _Z * _standard_error(bootstrap.replicates)


def _p_value(bootstrap: _Bootstrap) -> float:
    if math.isnan(bootstrap.point) or bootstrap.replicates.size < 2:
        return UNDEFINED
    error = _standard_error(bootstrap.replicates)
    if error == 0.0:
        return 1.0 if bootstrap.point == 0.0 else UNDEFINED
    return math.erfc(abs(bootstrap.point) / (error * math.sqrt(2.0)))


def _checked_metric(metric: object) -> Callable[[np.ndarray, np.ndarray], float]:
    if not callable(metric):
        raise ValueError(
            f"metric must be callable as metric(y_true, y_score), got {metric!r}; "
            f"a metric taking more than the two arrays is bound with "
            f"functools.partial before it is passed"
        )
    return metric


def _checked_metrics(
    metric: object, count: int
) -> tuple[Callable[[np.ndarray, np.ndarray], float], ...]:
    if callable(metric):
        return (_checked_metric(metric),) * count
    try:
        given = tuple(metric)
    except TypeError:
        raise ValueError(
            f"metric must be callable as metric(y_true, y_score), or a sequence "
            f"of {count} such callables, one per fold, got {metric!r}"
        ) from None
    if len(given) != count:
        raise ValueError(
            f"metric holds {len(given)} callables and there are {count} folds; a "
            f"sequence of metrics carries one per fold"
        )
    return tuple(_checked_metric(item) for item in given)


def _checked_kind(target_kind: object) -> TargetKind:
    if not isinstance(target_kind, TargetKind):
        raise ValueError(
            f"target_kind must be a TargetKind, got {target_kind!r}; it says how "
            f"y_true is read and how the resampling is stratified"
        )
    return target_kind


def _checked_pair(
    y_true: object, y_score: object, name: str, target_kind: TargetKind
) -> tuple[np.ndarray, np.ndarray]:
    check = (
        check_pair if target_kind is TargetKind.BINARY else check_continuous_pair
    )
    try:
        return check(y_true, y_score)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error


def confidence_interval(
    y_true: object,
    y_score: object,
    metric: object,
    *,
    fold: object = None,
    target_kind: TargetKind = TargetKind.BINARY,
) -> Interval:
    kind = _checked_kind(target_kind)
    targets, scores = _checked_pair(y_true, y_score, "y_score", kind)
    folds = _folds(targets, (scores,), fold, kind)
    bootstrap = _bootstrap(_checked_metrics(metric, len(folds)), folds)
    return Interval(bootstrap.point, _half_width(bootstrap), bootstrap.undefined_fraction)


def confidence_intervals(
    y_true: object,
    y_score: object,
    metrics: dict[str, object],
    *,
    fold: object = None,
    target_kind: TargetKind = TargetKind.BINARY,
) -> dict[str, Interval]:
    kind = _checked_kind(target_kind)
    targets, scores = _checked_pair(y_true, y_score, "y_score", kind)
    folds = _folds(targets, (scores,), fold, kind)
    sets = {
        name: _checked_metrics(metric, len(folds))
        for name, metric in metrics.items()
    }
    points = {
        name: _statistic(
            [
                _value(metric, fold_.targets, fold_.columns)
                for metric, fold_ in zip(functions, folds)
            ]
        )
        for name, functions in sets.items()
    }
    measurable = {name: sets[name] for name in sets if not math.isnan(points[name])}
    replicates = _resample_many(measurable, folds) if measurable else {}

    intervals: dict[str, Interval] = {}
    for name in sets:
        point = points[name]
        if name not in measurable:
            intervals[name] = Interval(point, UNDEFINED, UNDEFINED)
            continue
        values = replicates[name]
        defined = ~np.isnan(values)
        undefined_fraction = 1.0 - float(defined.mean())
        kept = (
            values[defined] if undefined_fraction <= UNDEFINED_TOLERANCE else np.empty(0)
        )
        bootstrap = _Bootstrap(point, kept, undefined_fraction)
        intervals[name] = Interval(point, _half_width(bootstrap), undefined_fraction)
    return intervals


def paired_difference(
    y_true: object,
    y_score_a: object,
    y_score_b: object,
    metric: object,
    *,
    fold: object = None,
    target_kind: TargetKind = TargetKind.BINARY,
) -> Comparison:
    kind = _checked_kind(target_kind)
    targets, scores_a = _checked_pair(y_true, y_score_a, "y_score_a", kind)
    _, scores_b = _checked_pair(y_true, y_score_b, "y_score_b", kind)
    folds = _folds(targets, (scores_a, scores_b), fold, kind)
    bootstrap = _bootstrap(_checked_metrics(metric, len(folds)), folds)
    half_width = _half_width(bootstrap)
    p_value = UNDEFINED if math.isnan(half_width) else _p_value(bootstrap)
    return Comparison(
        bootstrap.point, half_width, p_value, bootstrap.undefined_fraction
    )


def _p_values(p_values: object) -> np.ndarray:
    values = np.asarray(p_values)
    if values.ndim != 1:
        raise ValueError(
            f"p_values must be a one dimensional array, its shape is {values.shape}"
        )
    if values.dtype.kind not in "biuf":
        raise ValueError(f"p_values must hold numbers, its dtype is {values.dtype}")
    if values.size == 0:
        raise ValueError(
            "p_values is empty; a family with no comparison in it is an empty "
            "selection upstream, not a correction with no value"
        )
    values = values.astype("float64", copy=True)
    present = ~np.isnan(values)
    outside = values[present][(values[present] < 0.0) | (values[present] > 1.0)]
    if outside.size:
        raise ValueError(
            f"p_values must lie in the unit interval, {outside.size} do not, the "
            f"first being {outside[0]}"
        )
    return values


def _adjusted(p_values: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _p_values(p_values)
    tested = np.flatnonzero(~np.isnan(values))
    order = tested[np.argsort(values[tested], kind="stable")]
    return values, order, values[order]


def _family_size(family_size: int | None, count: int) -> int:
    if family_size is None:
        return count
    if family_size < count:
        raise ValueError(
            f"family_size must be at least the number of p values given, got "
            f"{family_size} for {count}"
        )
    return int(family_size)


def holm_adjusted(p_values: object, *, family_size: int | None = None) -> np.ndarray:
    values, order, ascending = _adjusted(p_values)
    count = order.size
    size = _family_size(family_size, count)
    if count:
        scaled = ascending * (size - np.arange(count))
        values[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return values


def benjamini_hochberg_adjusted(
    p_values: object, *, family_size: int | None = None
) -> np.ndarray:
    values, order, ascending = _adjusted(p_values)
    count = order.size
    size = _family_size(family_size, count)
    if count:
        scaled = ascending * size / (np.arange(count) + 1.0)
        values[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return values


def _log_choose(total: int, taken: int) -> float:
    return math.lgamma(total + 1.0) - math.lgamma(taken + 1.0) - math.lgamma(total - taken + 1.0)


def fisher_exact_p(table: object) -> float:
    values = np.asarray(table)
    if values.shape != (2, 2):
        raise ValueError(f"fisher_exact_p: the table must be two by two, its shape is {values.shape}")
    if values.dtype.kind not in "iu":
        if values.dtype.kind != "f" or not np.all(np.floor(values) == values):
            raise ValueError(f"fisher_exact_p: the counts must be integers, got {values.tolist()}")
    counts = values.astype("int64")
    if (counts < 0).any():
        raise ValueError(f"fisher_exact_p: the counts must be non-negative, got {counts.tolist()}")
    a, b = int(counts[0, 0]), int(counts[0, 1])
    c, d = int(counts[1, 0]), int(counts[1, 1])
    row = a + b
    column = a + c
    total = a + b + c + d
    if row == 0 or column == 0 or row == total or column == total:
        return 1.0
    denominator = _log_choose(total, column)

    def log_probability(x: int) -> float:
        return _log_choose(row, x) + _log_choose(total - row, column - x) - denominator

    observed = log_probability(a)
    ceiling = observed + math.log1p(1e-7)
    low = max(0, column - (total - row))
    high = min(row, column)
    p_value = 0.0
    for x in range(low, high + 1):
        value = log_probability(x)
        if value <= ceiling:
            p_value += math.exp(value)
    return min(1.0, p_value)


def _kept(keep: object, count: int, name: str) -> np.ndarray:
    if keep is None:
        return np.ones(count, dtype=bool)
    mask = np.asarray(keep)
    if mask.dtype != bool or mask.shape != (count,):
        raise ValueError(
            f"{name} must be one boolean per gene, y_true holds {count} values and "
            f"{name} has shape {mask.shape} and dtype {mask.dtype}"
        )
    if not mask.any():
        raise ValueError(f"{name} keeps no gene; an empty view is an empty selection upstream")
    return mask


class _View(NamedTuple):

    scores: np.ndarray
    folds: np.ndarray
    keep: np.ndarray
    fold_values: tuple[int, ...]


def _view_statistic(
    metric: Callable[[np.ndarray, np.ndarray], float],
    targets: np.ndarray,
    view: _View,
    take: np.ndarray | None,
) -> float:
    labels = targets if take is None else targets[take]
    scores = view.scores if take is None else view.scores[take]
    folds = view.folds if take is None else view.folds[take]
    keep = view.keep if take is None else view.keep[take]
    values = []
    for value in view.fold_values:
        inside = keep & (folds == value)
        if not inside.any():
            return UNDEFINED
        values.append(_value(metric, labels[inside], [scores[inside]]))
    return _statistic(values)


def paired_view_differences(
    y_true: object,
    y_score_a: object,
    y_score_b: object,
    metrics: dict[str, object],
    *,
    fold_a: object = None,
    fold_b: object = None,
    keep_a: object = None,
    keep_b: object = None,
) -> dict[str, Comparison]:
    targets, scores_a = _checked_pair(y_true, y_score_a, "y_score_a", TargetKind.BINARY)
    _, scores_b = _checked_pair(y_true, y_score_b, "y_score_b", TargetKind.BINARY)
    count = targets.size
    folds_a = _fold_identifier(fold_a, count)
    folds_b = _fold_identifier(fold_b, count)
    kept_a = _kept(keep_a, count, "keep_a")
    kept_b = _kept(keep_b, count, "keep_b")
    named = {name: _checked_metric(metric) for name, metric in metrics.items()}
    order = np.lexsort((scores_b, scores_a, targets, kept_b, kept_a, folds_b, folds_a))
    laid = targets[order].astype("float64")
    cell_keys = np.column_stack(
        [folds_a[order], folds_b[order], kept_a[order], kept_b[order], targets[order]]
    )
    changes = np.flatnonzero(np.any(cell_keys[1:] != cell_keys[:-1], axis=1)) + 1
    strata = tuple(int(size) for size in np.diff(np.concatenate([[0], changes, [count]])))
    views = tuple(
        _View(
            scores[order],
            folds[order],
            kept[order],
            tuple(int(value) for value in np.unique(folds[kept])),
        )
        for scores, folds, kept in ((scores_a, folds_a, kept_a), (scores_b, folds_b, kept_b))
    )

    def difference(name: str, take: np.ndarray | None) -> float:
        first = _view_statistic(named[name], laid, views[0], take)
        if math.isnan(first):
            return UNDEFINED
        second = _view_statistic(named[name], laid, views[1], take)
        return UNDEFINED if math.isnan(second) else first - second

    points = {name: difference(name, None) for name in named}
    measurable = [name for name in named if not math.isnan(points[name])]
    replicates = {name: np.empty(RESAMPLE_COUNT, dtype="float64") for name in measurable}
    if measurable:
        generator = np.random.default_rng(SEED)
        for index in range(RESAMPLE_COUNT):
            take = _draw(generator, strata)
            for name in measurable:
                replicates[name][index] = difference(name, take)
    comparisons: dict[str, Comparison] = {}
    for name in named:
        point = points[name]
        if name not in replicates:
            comparisons[name] = Comparison(point, UNDEFINED, UNDEFINED, UNDEFINED)
            continue
        values = replicates[name]
        defined = ~np.isnan(values)
        undefined_fraction = 1.0 - float(defined.mean())
        kept = values[defined] if undefined_fraction <= UNDEFINED_TOLERANCE else np.empty(0)
        bootstrap = _Bootstrap(point, kept, undefined_fraction)
        half_width = _half_width(bootstrap)
        p_value = UNDEFINED if math.isnan(half_width) else _p_value(bootstrap)
        comparisons[name] = Comparison(point, half_width, p_value, undefined_fraction)
    return comparisons
