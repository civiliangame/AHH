"""Tools for tuning pairs of scalar thresholds to trade off sensitivity, specificity, and indeterminate rate.

See `IndetSnSpArray` and subclass docstrings for details.

"""
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional
from typing_extensions import Self  # in typing in python3.11

import numpy as np


def running_argmax_indices(a):
    """Return indices of a where the value is larger than all previous values.

    >>> running_argmax_indices([1, 0, 3, 4, 4, 2, 5, 7, 1])
    array([0, 2, 3, 6, 7])

    """
    m = np.maximum.accumulate(a)
    return np.flatnonzero(np.r_[True, m[:-1] < m[1:]])


def pareto_2d_indices(x, y):
    """Compute indices of the Pareto frontier maximizing x and y, sorted in increasing x and decreasing y.

    e.g. the Pareto frontier of the point set below is [A, G]

    B A
     C
    E D
    F  G
     H

    >>> u = [2, 0, 1, 2, 0, 0, 3, 1]
    >>> v = [4, 4, 3, 2, 2, 1, 1, 0]
    >>> pareto_2d_indices(np.array(u), np.array(v))
    array([0, 6])

    """
    sort_indices = np.lexsort((-x, -y))  # last element is primary sort key
    return sort_indices[running_argmax_indices(x[sort_indices])]


def midpoints_with_infs(x):
    """Return the midpoints between the sorted unique elements of x, along with +/-inf."""
    unique_scores = np.unique(np.r_[-np.inf, x, np.inf])
    return (unique_scores[1:] + unique_scores[:-1]) / 2


def kde_disc_mass(
    data: np.ndarray,
    edges: np.ndarray,
    bandwidth: float,
    weights: Optional[np.ndarray] = None,
):
    """Perform Kernel Density Estimation (KDE) on data & weights, then compute the probability mass between edges."""
    import scipy
    z_score = (edges[:, None] - data[None, :]) / bandwidth
    component_cdfs = scipy.stats.norm.cdf(z_score)
    if weights is None:
        weights = np.ones_like(data)
    cdf = np.dot(component_cdfs, weights / weights.sum())
    return np.diff(cdf)


class BinaryLabeledScores(NamedTuple):
    """An array of numeric scores along with associated 0/1 ground truth and optional numeric weights."""

    y_score: np.ndarray
    y_true: np.ndarray
    weights: Optional[np.ndarray] = None

    def smooth(
        self, num_points: int, bandwidth: float, padding_bandwidths: float = 5.0
    ) -> "BinaryLabeledScores":
        """KDE-smooth positive and negative scores separately and discretize each to equally spaced weighted points.

        Args:
            num_points: number of points to use each for positive and negative score discretizations
            bandwidth: bandwidth of kernel density estimation, i.e. standard deviation of noise to be added
            padding_bandwidths: number of bandwidths to extend past lowest and highest scores when selecting
                discretization endpoints

        Returns:
            `BinaryLabeledScores` object representing the smoothed and re-discretized weighted labeled scores

        """
        pos = self.y_true == 1
        neg = self.y_true == 0
        if self.weights is not None:
            pos_weights = self.weights[pos]
            neg_weights = self.weights[neg]
        else:
            pos_weights = None
            neg_weights = None
        padding = padding_bandwidths * bandwidth
        all_points = np.linspace(
            self.y_score.min() - padding,
            self.y_score.max() + padding,
            2 * num_points + 1,
        )
        edges = all_points[::2]
        centers = all_points[1::2]
        pos_kde_weights = kde_disc_mass(
            self.y_score[pos], edges, bandwidth, pos_weights
        )
        neg_kde_weights = kde_disc_mass(
            self.y_score[neg], edges, bandwidth, neg_weights
        )
        return BinaryLabeledScores(
            y_true=np.r_[np.zeros_like(centers), np.ones_like(centers)],
            y_score=np.r_[centers, centers],
            weights=np.r_[neg_kde_weights, pos_kde_weights],
        )

    def indet_sn_sp_array(self) -> "IndetSnSpArray":
        """Build `IndetSnSpArray`."""
        return IndetSnSpArray.build(**self._asdict())


def fake_vectorized_binom_ci(
    k: np.ndarray, n: np.ndarray, p: float | np.ndarray = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binomial confidence intervals on arrays of parameters inefficiently."""
    import scipy
    k, n, p = np.broadcast_arrays(k, n, p)
    # If speed is needed this can be rewritten with the statsmodels package, which is vectorized.
    flat_out = [
        scipy.stats.binomtest(k_, n_).proportion_ci(p_)
        for k_, n_, p_ in zip(k.flatten(), n.flatten(), p.flatten())
    ]
    low = np.array([ci.low for ci in flat_out]).reshape(k.shape)
    high = np.array([ci.high for ci in flat_out]).reshape(k.shape)
    return low, high


@dataclass
class IndetSnSpArray:
    """An array of metrics at different lower and upper threshold values.

    This class and subclasses are for selecting pairs of model thresholds based on sensitivity,
    specificity, and indeterminate rate. Throughout we assume scores are scalars and ground truth is binary.

    Each member `lower_thresh`, `upper_thresh`, `sn`, `sp`, and `indet_frac` must be a numpy array, and they all must
    have the same shape. Corresponding entries of these arrays specify a pair of thresholds and the metrics when a
    common dataset is evaluated using those thresholds. The thresholding logic is that scores less than the lower
    threshold count as negative outputs, scores greater than or equal to the upper threshold count as positive outputs,
    and scores in between are indeterminate outputs.

    Indeterminate fraction is defined as the proportion of scores in between the two thresholds. All other
    metrics are interpreted as conditioned on the scores not being indeterminate. For example, sensitivity
    is defined as usual as (true positives) / (total positives) *except that examples with indeterminate
    scores do not count towards the numerator or the denominator*.

    """

    lower_thresh: np.ndarray
    upper_thresh: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray
    indet: np.ndarray
    weighted: bool = False
    min_weight: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        self.sn = self.tp / np.maximum(self.tp + self.fn, self.min_weight)
        self.sp = self.tn / np.maximum(self.tn + self.fp, self.min_weight)
        self.ppv = self.tp / np.maximum(self.tp + self.fp, self.min_weight)
        self.npv = self.tn / np.maximum(self.tn + self.fn, self.min_weight)
        total = self.indet + self.fn + self.fp + self.tn + self.tp
        self.indet_frac = self.indet / np.maximum(total, self.min_weight)
        for attr in ("sn", "sp", "ppv", "npv", "indet_frac"):
            value = getattr(self, attr)
            if value.size:
                if value.max() > 1 + self.eps:
                    raise ValueError(
                        f"Numerical precision issues produced invalid value {attr} = {value.max()}."
                    )
                if value.min() < -self.eps:
                    raise ValueError(
                        f"Numerical precision issues produced invalid value {attr} = {value.min()}."
                    )
            setattr(self, attr, np.clip(value, 0.0, 1.0))

    @property
    def min_sn_sp(self):
        return np.minimum(self.sn, self.sp)

    @classmethod
    def build(
        cls,
        lower_thresh: Optional[np.ndarray] = None,
        upper_thresh: Optional[np.ndarray] = None,
        *,
        y_true: np.ndarray,
        y_score: np.ndarray,
        weights: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> Self:
        """Find `IndetSnSpArray` values for given truth and scores as thresholds vary (à la sklearn.metrics.roc_curve).

        The output object contains arrays for `sn`, `sp`, `indet_frac`, `lower_thresh`, and `upper_thresh`, all with the
        same shape. What these arrays contain and what their common shape is depends on the input as follows.

        If both lower_thresh and upper_thresh are provided, they must have the same shape and this method computes
        metrics at the pairs given by corresponding entries in these arrays. The common output shape will be the same as
        this common input shape.

        If only one set of thresholds is provided, this method computes metrics at all sorted pairs of these thresholds
        (along with +/- inf). If neither is provided, sort scores and allow thresholds between each pair (along with
        +/- inf). In both of these cases, the common output shape is a 1-d vector of length equal to the number of such
        pairs.

        """
        weights = weights if weights is not None else np.ones_like(y_true)
        y_true = y_true[weights > 0]
        y_score = y_score[weights > 0]
        weights = weights[weights > 0]

        # Find all threshes and include +/- inf so np.histogram does the right thing
        if lower_thresh is not None and upper_thresh is not None:
            threshes = np.unique(np.r_[-np.inf, lower_thresh, upper_thresh, np.inf])
            lower_indices = np.searchsorted(threshes, lower_thresh)
            upper_indices = np.searchsorted(threshes, upper_thresh)
        else:
            if lower_thresh is not None:
                threshes = np.unique(np.r_[-np.inf, lower_thresh, np.inf])
            elif upper_thresh is not None:
                threshes = np.unique(np.r_[-np.inf, upper_thresh, np.inf])
            else:
                unique_scores = np.unique(np.r_[-np.inf, y_score, np.inf])
                threshes = (unique_scores[1:] + unique_scores[:-1]) / 2
                threshes = np.unique(threshes)
            lower_indices, upper_indices = np.triu_indices(len(threshes))

        count_by_bin = np.histogram(y_score, bins=threshes, weights=weights)[0]
        pos_by_bin = np.histogram(y_score, bins=threshes, weights=y_true * weights)[0]
        count_by_thresh = np.pad(np.cumsum(count_by_bin), (1, 0))
        pos_by_thresh = np.pad(np.cumsum(pos_by_bin), (1, 0))
        tn_plus_fn = count_by_thresh[lower_indices]
        total_minus_tp_minus_fp = count_by_thresh[upper_indices]
        tp_plus_fp = count_by_thresh[-1] - total_minus_tp_minus_fp
        fn = pos_by_thresh[lower_indices]
        total_pos = pos_by_thresh[-1]  # last thresh is +inf
        tp = total_pos - pos_by_thresh[upper_indices]
        fp = tp_plus_fp - tp
        tn = tn_plus_fn - fn
        min_weight = weights.min()
        indet = total_minus_tp_minus_fp - tn_plus_fn
        return cls(
            lower_thresh=threshes[lower_indices],
            upper_thresh=threshes[upper_indices],
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            indet=indet,
            weighted=not all(weights == 1.0),
            min_weight=min_weight,
            eps=eps,
        )

    def eval(
        self,
        *,
        y_true,
        y_score,
        weights: Optional[np.ndarray] = None,
    ) -> "IndetSnSpArray":
        """Evaluate the given data on the thresholds of `self`."""
        return IndetSnSpArray.build(
            lower_thresh=self.lower_thresh,
            upper_thresh=self.upper_thresh,
            y_true=y_true,
            y_score=y_score,
            weights=weights,
        )

    def __getitem__(self, item) -> "IndetSnSpArray":
        """Extract a subarray with numpy-style indexing."""
        return IndetSnSpArray(
            lower_thresh=self.lower_thresh[item],
            upper_thresh=self.upper_thresh[item],
            tp=self.tp[item],
            fp=self.fp[item],
            fn=self.fn[item],
            tn=self.tn[item],
            indet=self.indet[item],
            weighted=self.weighted,
            min_weight=self.min_weight,
            eps=self.eps,
        )

    def __add__(self, other: "IndetSnSpArray") -> "IndetSnSpArray":
        if not isinstance(other, IndetSnSpArray):
            raise TypeError(f"Cannot add {type(other)} to IndetSnSpArray.")
        tp = self.tp + other.tp
        if np.array_equal(self.lower_thresh, other.lower_thresh):
            lower_thresh = self.lower_thresh
        else:
            lower_thresh = np.nan * np.ones_like(tp)
        if np.array_equal(self.upper_thresh, other.upper_thresh):
            upper_thresh = self.upper_thresh
        else:
            upper_thresh = np.nan * np.ones_like(tp)
        return IndetSnSpArray(
            lower_thresh=lower_thresh,
            upper_thresh=upper_thresh,
            tp=tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
            indet=self.indet + other.indet,
            weighted=self.weighted or other.weighted,
            min_weight=min(self.min_weight, other.min_weight),
            eps=self.eps,
        )

    def confidence_interval_bound(self, p: float = 0.95) -> "IndetSnSpArray":
        """Compute two-sided confidence interval bounds: upper for indet_frac and lower for other metrics."""
        if self.weighted:
            raise NotImplementedError(
                "Confidence intervals only implemented for unweighted confusion matrices."
            )
        copy = IndetSnSpArray(**asdict(self))
        copy.sn, _ = fake_vectorized_binom_ci(self.tp, self.tp + self.fn, p=p)
        copy.sp, _ = fake_vectorized_binom_ci(self.tn, self.tn + self.fp, p=p)
        copy.ppv, _ = fake_vectorized_binom_ci(self.tp, self.fp + self.tp, p=p)
        copy.npv, _ = fake_vectorized_binom_ci(self.tn, self.tn + self.fn, p=p)
        _, copy.indet_frac = fake_vectorized_binom_ci(
            self.indet, self.tp + self.fn + self.fp + self.tn + self.indet, p=p
        )
        return copy

    def roc_curve(self, indet_budget=0.0) -> "IndetRocCurve":
        """Compute ROC curve with indeterminate budget, sorted by increasing sn and decreasing sp.

        Restrict `self` to Pareto-optimal pairs (sn, sp) for which `indet_frac <= indet_budget`. Other points are worse
        than the points on the curve in the sense of having worse Sn, worse Sp, or not meeting the indeterminate budget.

        """
        within_budget = self[self.indet_frac <= indet_budget]
        frontier = pareto_2d_indices(within_budget.sn, within_budget.sp)
        return IndetRocCurve(**asdict(within_budget[frontier]))

    def sn_eq_sp_graph(self) -> "IndetSnEqSpGraph":
        """Compute sn=sp as a function of indet_frac, returning both sorted in increasing order.

        Method: restrict to Pareto-optimal pairs (s, indet_frac) where s = min(sn, sp).

        Pareto-optimality means that if (s, indet_frac) is in the output, there is no point (sn', sp', indet_frac') in
        the input with indet_frac' <= indet_frac and sn', sp' > s. In other words, s is the maximum value such that
        the quadrant { sn, sp >= s} intersects `self.roc_curve(indet_frac)`. This maximum occurs where the ROC curve
        intersects the diagonal, up to an error bounded by the distance between points on the ROC curve.

        """
        frontier = pareto_2d_indices(self.min_sn_sp, -self.indet_frac)
        return IndetSnEqSpGraph(**asdict(self[frontier]))


class IndetRocCurve(IndetSnSpArray):
    """Sn, Sp achievable within some indeterminate budget and associated lower and upper thresholds.

    `sn` is assumed to be sorted in increasing order and `sp` decreasing.

    """

    def sn_eq_sp(self) -> IndetSnSpArray:
        """Locate the point on the ROC curve closest to the diagonal"""
        return self[np.argmax(self.min_sn_sp)]

    def auc(self) -> float:
        """Compute the area under the ROC curve."""
        # `auc` does not automatically include the trivial points (0, 1) and (1, 0)
        # and will underestimate the AUC if these are not explicitly added
        from sklearn.metrics import auc
        return auc(1 - np.r_[1.0, self.sp, 0.0], np.r_[0.0, self.sn, 1.0])

    @classmethod
    def build(
        cls,
        thresh=None,
        *,
        y_true: np.ndarray,
        y_score: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Self:
        """Build an indeterminate=0 ROC curve in n log n time (vs n**2 for IndetSnSpArray.build().roc_curve())."""
        if thresh is None:
            thresh = midpoints_with_infs(y_score)[
                ::-1
            ]  # reverse for proper output sorting
        issa = IndetSnSpArray.build(
            lower_thresh=thresh,
            upper_thresh=thresh,
            y_true=y_true,
            y_score=y_score,
            weights=weights,
        )
        return cls(**asdict(issa))


class IndetSnEqSpGraph(IndetSnSpArray):
    """Sn=Sp achievable as a function of indeterminate budget and associated lower and upper thresholds.

    Both min(self.sn, self.sp) and self.indet_frac are assumed to be sorted in non-decreasing order.

    """

    def at_budget(self, indet_budget: float = 0.0) -> IndetSnSpArray:
        """Locate the best point on the graph within the given budget."""
        return self[np.searchsorted(self.indet_frac, indet_budget, side="right") - 1]
