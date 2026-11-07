# Copyright (C) 2026 by The Biopython Contributors
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Conditional substitution statistics on a fixed rooted phylogenetic tree.

Given a rooted tree with branch lengths, a discrete-character alignment, a
continuous-time Markov chain (CTMC) rate matrix ``Q``, a root state
distribution and a discrete mixture of substitution-rate categories, this
module computes, for every branch and every alignment site:

- the conditional expected number of substitutions for each *directed*
  state pair ``i -> j`` (off-diagonal entries of the count matrix; the
  diagonal is identically zero), and
- the conditional expected dwell time spent in each state.

The expectations are taken over the full hidden substitution history along
each branch conditional on the observed alignment, so substitutions that
cancel out at the branch endpoints (hidden back-and-forth changes) are fully
counted. This is the deterministic, expectation-based form of ancestral
substitution mapping (Dutheil et al., 2005; Minin and Suchard, 2008): no
external program and no stochastic simulation are used.

Mathematical background
-----------------------
For a branch of length ``t`` evolving at rate ``r`` with generator ``Q``,
let ``P(s) = exp(r Q s)``. For an additive statistic with reward matrix
``B`` (``B[i, j]`` is accumulated while jumping from state ``i`` to state
``j``; dwell-time rewards use a diagonal ``B``), the integral kernel

    I^B(t) = integral_0^t P(s) B P(t - s) ds

gives, after contraction with the forward partial likelihood at the parent
and the backward partial likelihood at the child, the conditional
expectation on the branch. For directed substitution counts ``i -> j`` the
reward is ``B[i, j] = r * Q[i, j]``; for dwell time in state ``i`` the
reward is ``B[i, i] = 1``. With ``Q = V diag(lambda) V**-1`` the kernel is
evaluated analytically as

    I^B(t) = V ( Phi * (V**-1 B V) ) V**-1,

where ``Phi[m, l]`` is the time integral of the paired exponential factors
``exp(r lambda_m s) exp(r lambda_l (t - s))``, computed with a stable
``expm1`` formula. Time-reversible generators are diagonalized through a
symmetric similarity transform (real, numerically stable); arbitrary
non-reversible generators use the general (possibly complex)
eigendecomposition.

Partial likelihood vectors are normalized at every node of both the
downward (pruning) and the upward passes, with the normalization factors
accumulated in log space, so very small but non-zero site likelihoods are
handled without underflow. If a site has exactly zero likelihood, the
conditional statistics are not defined and a ``ZeroLikelihoodError`` is
raised.

References
----------
Dutheil, J. Y., Pupko, T., Jean-Marie, A., and Galtier, N. (2005).
A model-based approach for detecting coevolving positions in a molecule.
Molecular Biology and Evolution, 22(9), 1919-1928.

Minin, V. N. and Suchard, M. A. (2008). Counting labeled transitions in
continuous-time Markov models of evolution. Journal of Mathematical
Biology, 56(3), 391-412.
"""

import numpy as np

from Bio.Data.IUPACData import ambiguous_dna_values
from Bio.Data.IUPACData import ambiguous_rna_values
from Bio.Data.IUPACData import protein_letters


class ZeroLikelihoodError(ValueError):
    """Raised when a site has zero likelihood.

    Conditional substitution counts and dwell times are expectations taken
    under the posterior distribution of substitution histories. When the
    site likelihood is zero, the conditioning event is impossible and the
    conditional expectations are therefore undefined.
    """


class BranchSubstitutionStats:
    """Conditional substitution statistics on a single branch.

    The branch runs from ``parent`` to ``clade`` (the child node) and has
    length ``branch_length``.

    Attributes
    ----------
    parent : Clade
        Parent node of the branch.
    clade : Clade
        Child node of the branch; the branch is the incoming branch of this
        clade.
    branch_length : float
        Length of the branch.
    states : tuple
        State labels, in the order used by the rate matrix rows/columns.
    counts : numpy.ndarray
        Array of shape ``(n_sites, n_states, n_states)`` with the
        conditional expected number of substitutions. Entry
        ``counts[s, i, j]`` is the expected number of ``states[i]`` ->
        ``states[j]`` substitutions at site ``s`` on this branch. The
        diagonal is zero.
    dwell : numpy.ndarray
        Array of shape ``(n_sites, n_states)`` with the conditional expected
        time spent in each state at each site on this branch. Each row sums
        to the branch length.
    """

    __slots__ = ("parent", "clade", "branch_length", "states", "counts", "dwell")

    def __init__(self, parent, clade, branch_length, states, counts, dwell):
        """Initialize a branch statistics record."""
        self.parent = parent
        self.clade = clade
        self.branch_length = float(branch_length)
        self.states = tuple(states)
        self.counts = counts
        self.dwell = dwell

    def sum(self, sites=None):
        """Return total counts and dwell times over selected sites.

        :param sites: site indices (int, slice, integer array or boolean
            mask); ``None`` means all sites.
        :returns: tuple ``(counts, dwell)`` with arrays of shape
            ``(n_states, n_states)`` and ``(n_states,)``.
        """
        if sites is None:
            return self.counts.sum(axis=0), self.dwell.sum(axis=0)
        if isinstance(sites, (int, np.integer)):
            return self.counts[sites], self.dwell[sites]
        return self.counts[sites].sum(axis=0), self.dwell[sites].sum(axis=0)

    @property
    def name(self):
        """Name of the child clade (or ``None`` if unnamed)."""
        return self.clade.name

    def __repr__(self):
        return (
            f"BranchSubstitutionStats(parent={self.parent.name!r}, "
            f"clade={self.clade.name!r}, branch_length={self.branch_length:g}, "
            f"n_sites={self.counts.shape[0]}, n_states={self.counts.shape[1]})"
        )


class ConditionalSubstitutionStats:
    """Per-branch conditional substitution statistics for one tree.

    Iterating over the object yields ``BranchSubstitutionStats`` records in
    pre-order (parent branches before child branches). Individual branches
    can be retrieved by their child clade object or by the child clade's
    name using indexing, e.g. ``result[clade]`` or ``result["taxonA"]``.

    Attributes
    ----------
    tree : Tree
        The input tree (not modified).
    states : tuple
        State labels in rate-matrix order.
    rates : numpy.ndarray
        Rate multipliers of the discrete rate mixture.
    rate_weights : numpy.ndarray
        Prior mixture weights (normalized to sum one).
    site_log_likelihoods : numpy.ndarray
        Log likelihood of each alignment site under the mixture model.
    rate_posteriors : numpy.ndarray
        Array of shape ``(n_sites, n_rates)`` with the posterior
        probability of each rate category at each site.
    branches : list
        ``BranchSubstitutionStats`` records, one per branch (the root has
        no incoming branch and is not included).
    """

    __slots__ = (
        "tree",
        "states",
        "rates",
        "rate_weights",
        "site_log_likelihoods",
        "rate_posteriors",
        "branches",
        "_by_clade",
        "_by_name",
    )

    def __init__(
        self,
        tree,
        states,
        rates,
        rate_weights,
        site_log_likelihoods,
        rate_posteriors,
        branches,
    ):
        """Initialize a conditional substitution statistics result."""
        self.tree = tree
        self.states = tuple(states)
        self.rates = np.asarray(rates, dtype=float)
        self.rate_weights = np.asarray(rate_weights, dtype=float)
        self.site_log_likelihoods = np.asarray(site_log_likelihoods, dtype=float)
        self.rate_posteriors = np.asarray(rate_posteriors, dtype=float)
        self.branches = list(branches)
        self._by_clade = {id(b.clade): b for b in self.branches}
        self._by_name = {
            b.clade.name: b for b in self.branches if b.clade.name is not None
        }

    @property
    def log_likelihood(self):
        """Total log likelihood summed over all sites."""
        return float(self.site_log_likelihoods.sum())

    def __getitem__(self, key):
        """Look up a branch by child clade object or child clade name."""
        if isinstance(key, str):
            return self._by_name[key]
        return self._by_clade[id(key)]

    def __contains__(self, key):
        """Check whether a branch (by child clade or name) is present."""
        if isinstance(key, str):
            return key in self._by_name
        return id(key) in self._by_clade

    def get(self, key, default=None):
        """Return the branch record for ``key``, or ``default`` if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        """Iterate over branch records in pre-order."""
        return iter(self.branches)

    def __len__(self):
        """Return the number of branches."""
        return len(self.branches)

    def _site_index(self, sites):
        if sites is None:
            return slice(None)
        return sites

    def total_counts(self, sites=None):
        """Sum the directed substitution counts over all branches.

        :param sites: optional site index/array/mask restriction.
        :returns: array of shape ``(n_states, n_states)``.
        """
        idx = self._site_index(sites)
        return sum(b.counts[idx].sum(axis=0) for b in self.branches)

    def total_dwell(self, sites=None):
        """Sum the state dwell times over all branches.

        :param sites: optional site index/array/mask restriction.
        :returns: array of shape ``(n_states,)``.
        """
        idx = self._site_index(sites)
        return sum(b.dwell[idx].sum(axis=0) for b in self.branches)

    def __repr__(self):
        return (
            f"ConditionalSubstitutionStats(n_branches={len(self.branches)}, "
            f"n_sites={len(self.site_log_likelihoods)}, "
            f"n_states={len(self.states)}, "
            f"n_rates={len(self.rates)}, "
            f"log_likelihood={self.log_likelihood:.6f})"
        )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _alignment_rows(alignment):
    """Extract ``{sequence name: aligned sequence string}`` (PRIVATE)."""
    rows = {}
    if hasattr(alignment, "_records"):
        # Bio.Align.MultipleSeqAlignment
        for record in alignment._records:
            if record.id in rows:
                raise ValueError(f"Duplicate sequence id in alignment: {record.id!r}")
            rows[record.id] = str(record.seq)
    elif hasattr(alignment, "coordinates") and hasattr(alignment, "sequences"):
        # Bio.Align.Alignment (integer indexing returns the gapped row)
        for index, sequence in enumerate(alignment.sequences):
            name = (
                getattr(sequence, "id", None)
                or getattr(sequence, "name", None)
                or str(index)
            )
            if name in rows:
                raise ValueError(f"Duplicate sequence id in alignment: {name!r}")
            rows[name] = str(alignment[index])
    else:
        # Last resort: an iterable of SeqRecord-like objects
        for record in alignment:
            if record.id in rows:
                raise ValueError(f"Duplicate sequence id in alignment: {record.id!r}")
            rows[record.id] = str(record.seq)
    return rows


def _validate_tree(tree):
    """Check the tree is rooted with branch lengths; return traversal info."""
    root = tree.root
    if not getattr(tree, "rooted", True):
        raise ValueError(
            "Conditional substitution statistics require a rooted tree; "
            "root the tree (e.g. with Tree.root_with_outgroup) first."
        )
    preorder = []
    postorder = []
    parent_of = {}
    # Iterative traversal: reversed pre-order is a valid child-before-parent
    # post-order for a tree.
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        preorder.append(node)
        parent_of[id(node)] = parent
        for child in reversed(node.clades):
            stack.append((child, node))
    for node in reversed(preorder):
        postorder.append(node)
    for node in preorder:
        if node is root:
            continue
        blen = node.branch_length
        if blen is None:
            name = f" {node.name!r}" if node.name else ""
            raise ValueError(f"Missing branch length on the branch to node{name}.")
        if blen < 0:
            name = f" {node.name!r}" if node.name else ""
            raise ValueError(f"Negative branch length on the branch to node{name}.")
        node.branch_length = float(blen)
    terminals = [node for node in preorder if not node.clades]
    return root, preorder, postorder, parent_of, terminals


def _parse_rate_matrix(rate_matrix):
    """Validate the instantaneous rate matrix and return a float array."""
    Q = np.asarray(rate_matrix, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] < 2:
        raise ValueError("rate_matrix must be a square array with at least 2 states.")
    n = Q.shape[0]
    offdiag = Q[~np.eye(n, dtype=bool)]
    if np.any(offdiag < -1e-10 * max(1.0, abs(offdiag).max())):
        raise ValueError(
            "Off-diagonal entries of the rate matrix must be non-negative."
        )
    row_sums = Q.sum(axis=1)
    scale = max(1.0, abs(Q).max())
    if np.any(np.abs(row_sums) > 1e-8 * scale):
        raise ValueError(
            "Rows of the rate matrix must sum to zero (instantaneous "
            f"generator); got row sums {row_sums!r}."
        )
    # Enforce exact row sums to cancel accumulated rounding.
    Q = Q.copy()
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def _stationary_distribution(Q):
    """Return the stationary (left null) vector of ``Q`` (PRIVATE)."""
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    order = np.argsort(np.abs(eigenvalues.real))
    vec = eigenvectors[:, order[0]]
    if abs(eigenvalues[order[0]]) > 1e-8 * max(1.0, abs(Q).max()):
        raise ValueError(
            "Could not determine a unique stationary distribution from the "
            "rate matrix; please supply root_distribution explicitly."
        )
    pi = vec.real
    pi = pi / pi.sum()
    if np.any(pi < -1e-8):
        raise ValueError(
            "The rate matrix has no non-negative stationary distribution; "
            "please supply root_distribution explicitly."
        )
    pi = np.clip(pi, 0.0, None)
    total = pi.sum()
    if total <= 0:
        raise ValueError(
            "Failed to compute a stationary distribution; please supply "
            "root_distribution explicitly."
        )
    return pi / total


def _parse_root_distribution(root_distribution, Q):
    """Validate/solve the root distribution and return a probability vector."""
    if root_distribution is None:
        return _stationary_distribution(Q)
    pi = np.asarray(root_distribution, dtype=float)
    if pi.shape != (Q.shape[0],):
        raise ValueError(
            "root_distribution must have one entry per state "
            f"(length {Q.shape[0]}), got shape {pi.shape}."
        )
    if np.any(pi < 0):
        raise ValueError("root_distribution entries must be non-negative.")
    total = pi.sum()
    if total <= 0:
        raise ValueError("root_distribution must sum to a positive value.")
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"root_distribution must sum to 1.0, got {total!r}.")
    return pi / total


def _parse_rate_mixture(rate_mixture):
    """Parse the discrete rate mixture into rates and normalized weights."""
    if rate_mixture is None:
        return np.array([1.0]), np.array([1.0])
    arr = np.asarray(rate_mixture, dtype=float)
    if arr.ndim == 1:
        rates = arr
        if rates.size == 0:
            raise ValueError("rate_mixture must contain at least one rate.")
        weights = np.full(rates.shape, 1.0 / rates.size)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        rates, weights = arr[:, 0], arr[:, 1]
    else:
        raise ValueError(
            "rate_mixture must be a sequence of rates or a sequence of "
            "(rate, weight) pairs."
        )
    if np.any(rates < 0) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Mixture rates and weights must be non-negative.")
    return rates, weights / weights.sum()


def _build_character_vectors(states, character_vectors):
    """Build a ``{character: weight vector}`` mapping (PRIVATE)."""
    n = len(states)
    vectors = {}

    def add(code, members, weight=1.0):
        vec = np.zeros(n)
        for member in members:
            try:
                vec[states.index(member)] = weight
            except ValueError:
                return  # code not applicable to this state alphabet
        vectors[code] = vec

    # Exact states (one-hot)
    for state in states:
        add(state, [state])

    state_set = set(states)
    if state_set == set("ACGT"):
        for code, members in ambiguous_dna_values.items():
            add(code, members)
        add("U", ["T"])  # tolerate RNA U in a DNA-state alignment
    elif state_set == set("ACGU"):
        for code, members in ambiguous_rna_values.items():
            add(code, members)
        add("T", ["U"])  # tolerate DNA T in an RNA-state alignment
    elif state_set == set(protein_letters):
        for code, members in (("B", "ND"), ("Z", "EQ"), ("J", "IL")):
            add(code, members)

    # Fully uninformative characters: gap, missing, unknown.
    for code in ("-", "?", ".", "X", "N", "*"):
        vectors[code] = np.ones(n)

    # User-supplied vectors take precedence over the built-in handling.
    if character_vectors:
        for code, value in character_vectors.items():
            if isinstance(value, str):
                # Interpret a string as a set of allowed state labels.
                vec = np.zeros(n)
                for member in value:
                    vec[states.index(member)] = 1.0
            else:
                arr = np.asarray(value, dtype=float)
                if arr.shape == (n,):
                    vec = arr
                else:
                    # Interpret an iterable of state labels.
                    vec = np.zeros(n)
                    for member in value:
                        vec[states.index(str(member))] = 1.0
            if np.any(vec < 0):
                raise ValueError(f"Observation weights for {code!r} are negative.")
            vectors[str(code)] = vec
    return vectors


def _observation_vectors(rows, terminals, n_sites, states, character_vectors):
    """Build leaf observation arrays of shape (n_tips, n_sites, n_states)."""
    n = len(states)
    cmap = _build_character_vectors(states, character_vectors)
    missing_rows = [node.name for node in terminals if node.name not in rows]
    if missing_rows:
        raise ValueError(
            "Alignment has no row for tree terminal(s): "
            + ", ".join(repr(name) for name in missing_rows)
        )
    extra_rows = sorted(set(rows) - {node.name for node in terminals})
    if extra_rows:
        raise ValueError(
            "Alignment contains row(s) not found as tree terminals: "
            + ", ".join(repr(name) for name in extra_rows)
        )
    obs = np.empty((len(terminals), n_sites, n))
    for tip_index, node in enumerate(terminals):
        seq = rows[node.name]
        if len(seq) != n_sites:
            raise ValueError(
                f"Alignment row {node.name!r} has length {len(seq)}, expected "
                f"{n_sites}."
            )
        for site, char in enumerate(seq):
            try:
                obs[tip_index, site] = cmap[char]
            except KeyError:
                raise ValueError(
                    f"Character {char!r} (site {site}, taxon {node.name!r}) is "
                    "not a valid state or ambiguity code for states "
                    f"{tuple(states)!r}; provide it through character_vectors."
                ) from None
    return obs


# ---------------------------------------------------------------------------
# CTMC linear algebra
# ---------------------------------------------------------------------------


def _generator_basis(Q, pi):
    """Return ``(eigenvalues, V, V_inv, is_complex)`` for ``Q`` (PRIVATE).

    Time-reversible generators are diagonalized through the symmetric
    similarity transform ``S = diag(sqrt(pi)) Q diag(1/sqrt(pi))``; other
    generators use a general (possibly complex) eigendecomposition.
    """
    scale = max(1.0, abs(Q).max())
    flux = pi[:, None] * Q
    tol = 1e-9 * scale * max(pi.max(), 1.0)
    flux_symmetric = bool(np.all(np.abs(flux - flux.T) <= tol))
    if flux_symmetric and bool(np.all(pi > 0.0)):
        sqrt_pi = np.sqrt(pi)
        symmetric = (sqrt_pi[:, None] * Q) / sqrt_pi[None, :]
        # Guarantee exact symmetry against round-off.
        symmetric = 0.5 * (symmetric + symmetric.T)
        eigenvalues, U = np.linalg.eigh(symmetric)
        V = U / sqrt_pi[:, None]
        V_inv = U.T * sqrt_pi[None, :]
        return eigenvalues, V, V_inv, False
    if np.all(np.abs(Q - Q.T) <= 1e-9 * scale):
        # A symmetric generator is unitarily diagonalizable regardless of
        # the root distribution (which may assign zero mass to states).
        eigenvalues, U = np.linalg.eigh(0.5 * (Q + Q.T))
        return eigenvalues, U, U.T.copy(), False
    eigenvalues, V = np.linalg.eig(Q)
    V_inv = np.linalg.inv(V)
    return eigenvalues, V, V_inv, True


def _transition_and_phi(eigenvalues, V, V_inv, rate, t):
    """Return ``P(rate*t)`` and the paired-exponential integral ``Phi``."""
    if t == 0.0:
        # Exact identity so that zero-length branches preserve structural
        # zero likelihoods instead of introducing eigendecomposition noise.
        n = V.shape[0]
        return np.eye(n), np.zeros((n, n), dtype=eigenvalues.dtype)
    x = rate * t
    diag = x * eigenvalues
    # Phi[m, l] = t * exp(x*lambda_l) * expm1(x*(lambda_m - lambda_l))
    #             / (x*(lambda_m - lambda_l))
    # The leading factor t is the limit of (e**(r d t) - 1)/(r d) as
    # lambda_m -> lambda_l.
    delta = diag[:, None] - diag[None, :]
    exp_l = np.exp(diag)
    ratio = np.ones_like(delta)
    near_zero = np.abs(delta) < 1e-6
    if np.any(~near_zero):
        d = delta[~near_zero]
        ratio[~near_zero] = np.expm1(d) / d
    # Near zero: expm1(d)/d -> 1 + d/2 + ..., use a 2nd order correction.
    if np.any(near_zero):
        d = delta[near_zero]
        ratio[near_zero] = 1.0 + 0.5 * d
    phi = t * exp_l[None, :] * ratio
    transition = (V * np.exp(diag)[None, :]) @ V_inv
    imag_tol = 1e-10 * max(1.0, abs(transition).max())
    if np.any(np.abs(transition.imag) > imag_tol):
        raise FloatingPointError(
            "Non-negligible imaginary part in the transition probability; "
            "check the rate matrix."
        )
    transition = transition.real
    # Transition probabilities must be non-negative up to round-off.
    transition = np.where((transition < 0.0) & (transition > -1e-10), 0.0, transition)
    return transition, phi


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def conditional_substitution_stats(
    tree,
    alignment,
    rate_matrix,
    root_distribution=None,
    rate_mixture=None,
    states=None,
    character_vectors=None,
):
    """Compute conditional substitution counts and dwell times per branch.

    :param tree: a rooted Bio.Phylo tree (e.g. parsed from Newick). Every
        non-root clade must have a non-negative ``branch_length``. The tree
        is not modified.
    :param alignment: a ``Bio.Align.MultipleSeqAlignment`` or
        ``Bio.Align.Alignment`` whose row ids match the terminal names of
        the tree exactly.
    :param rate_matrix: array-like of shape ``(n_states, n_states)`` with
        the instantaneous substitution rate matrix ``Q``. Off-diagonal
        entries must be non-negative and rows must sum to zero. The
        convention is the standard one used for branch lengths: normalize
        ``Q`` so that one unit of branch length corresponds to one expected
        substitution per site, and rescale branches through the rate
        mixture if needed.
    :param root_distribution: vector of length ``n_states`` with the state
        probabilities at the root. If ``None``, the stationary distribution
        of ``Q`` is used (requires an ergodic generator).
    :param rate_mixture: discrete rate heterogeneity. Either a sequence of
        rate multipliers (equal prior weights) or a sequence of
        ``(rate, weight)`` pairs. ``None`` means a single rate of 1.
    :param states: ordered state labels, e.g. ``"ACGT"``. Defaults to
        ``("0", "1", ...)``. The order defines the rows/columns of
        ``rate_matrix``, ``root_distribution`` and the result arrays.
    :param character_vectors: optional mapping of alignment characters to
        per-state observation weights (array-like of length ``n_states``)
        or to iterables of state labels. Overrides the built-in handling of
        IUPAC ambiguity/missing characters.
    :returns: a ``ConditionalSubstitutionStats`` object.
    :raises ZeroLikelihoodError: if any alignment site has exactly zero
        likelihood; conditional statistics are undefined at such sites.

    The method is self-contained (NumPy only): no external phylogenetic
        program is invoked.

    Examples
    --------
    >>> import numpy as np
    >>> from Bio.Phylo import BaseTree
    >>> from Bio.Seq import Seq
    >>> from Bio.SeqRecord import SeqRecord
    >>> from Bio.Align import MultipleSeqAlignment
    >>> from Bio.Phylo.SubstitutionStats import conditional_substitution_stats
    >>> root = BaseTree.Clade(branch_length=None)
    >>> leaf_a = BaseTree.Clade(branch_length=0.5, name="A")
    >>> leaf_b = BaseTree.Clade(branch_length=0.5, name="B")
    >>> root.clades = [leaf_a, leaf_b]
    >>> tree = BaseTree.Tree(root, rooted=True)
    >>> aln = MultipleSeqAlignment(
    ...     [
    ...         SeqRecord(Seq("AC"), id="A"),
    ...         SeqRecord(Seq("A-"), id="B"),
    ...     ]
    ... )
    >>> Q = [[-1.0, 1.0], [1.0, -1.0]]
    >>> result = conditional_substitution_stats(
    ...     tree, aln, Q, states="AC", root_distribution=[0.5, 0.5]
    ... )
    >>> len(result)
    2
    >>> # Dwell times on a branch sum to the branch length, at every site:
    >>> for branch in result:
    ...     assert np.allclose(branch.dwell.sum(axis=1), 0.5)
    ...
    """
    root, preorder, postorder, parent_of, terminals = _validate_tree(tree)
    rows = _alignment_rows(alignment)
    n_sites = len(next(iter(rows.values()))) if rows else 0
    if n_sites == 0:
        raise ValueError("The alignment contains no sites.")

    Q = _parse_rate_matrix(rate_matrix)
    n_states = Q.shape[0]
    if states is None:
        states = tuple(str(i) for i in range(n_states))
    else:
        states = tuple(states)
        if len(states) != n_states:
            raise ValueError(
                f"Got {len(states)} state labels but the rate matrix has "
                f"{n_states} states."
            )
        if len(set(states)) != n_states:
            raise ValueError("State labels must be unique.")
    pi = _parse_root_distribution(root_distribution, Q)
    rates, rate_weights = _parse_rate_mixture(rate_mixture)

    obs = _observation_vectors(rows, terminals, n_sites, states, character_vectors)

    eigenvalues, V, V_inv, complex_basis = _generator_basis(Q, pi)
    if complex_basis:
        V_c = V
        V_inv_c = V_inv
    else:
        V_c = V.astype(complex)
        V_inv_c = V_inv.astype(complex)

    # Per-rate, per-edge transition matrices and Phi kernels.
    edges = [node for node in preorder if node is not root]
    transitions = {}
    phis = {}
    for k, rate in enumerate(rates):
        for node in edges:
            P, phi = _transition_and_phi(
                eigenvalues, V, V_inv, rate, node.branch_length
            )
            transitions[(k, id(node))] = P
            phis[(k, id(node))] = phi

    # Downward (pruning) pass with per-node normalized partials and log
    # normalization factors.
    leaf_sums = obs.sum(axis=2)  # (n_tips, n_sites)
    if np.any(leaf_sums <= 0.0):
        tip_index_arr, site_arr = np.nonzero(leaf_sums <= 0.0)
        raise ZeroLikelihoodError(
            f"Site {int(site_arr[0])} has an all-zero observation vector "
            f"for taxon {terminals[int(tip_index_arr[0])].name!r}; site "
            "likelihood is zero, so conditional statistics are undefined."
        )
    leaf_partials = obs / leaf_sums[:, :, None]
    leaf_log_scales = np.log(leaf_sums)

    site_components = np.empty((len(rates), n_sites))
    rate_posteriors = np.empty((len(rates), n_sites))
    down_partials = [None] * len(rates)
    down_log_scales = [None] * len(rates)
    root_dots = np.empty((len(rates), n_sites))

    for k, rate in enumerate(rates):
        partials = {}
        log_scales = {}
        for tip, d, a in zip(terminals, leaf_partials, leaf_log_scales):
            partials[id(tip)] = d
            log_scales[id(tip)] = a
        for node in postorder:
            if node.is_terminal():
                continue
            g = np.ones((n_sites, n_states))
            child_scale = np.zeros(n_sites)
            for child in node.clades:
                P = transitions[(k, id(child))]
                g = g * (partials[id(child)] @ P.T)
                child_scale = child_scale + log_scales[id(child)]
            col_sums = g.sum(axis=1)
            # Sites where this category assigns zero likelihood to every
            # state carry zero messages; they are excluded later via the
            # category posterior rather than producing NaNs.
            valid = col_sums > 0.0
            d_node = np.zeros_like(g)
            d_node[valid] = g[valid] / col_sums[valid, None]
            scale_node = child_scale.copy()
            scale_node[valid] = scale_node[valid] + np.log(col_sums[valid])
            partials[id(node)] = d_node
            log_scales[id(node)] = scale_node
        down_partials[k] = partials
        down_log_scales[k] = log_scales
        root_dot = (partials[id(root)] * pi[None, :]).sum(axis=1)
        root_dots[k] = root_dot
        component = np.full(n_sites, -np.inf)
        positive = root_dot > 0.0
        component[positive] = log_scales[id(root)][positive] + np.log(
            root_dot[positive]
        )
        site_components[k] = component

    # Mixture log likelihood and posterior category weights (log-sum-exp).
    weighted = np.log(rate_weights)[:, None] + site_components
    any_finite = np.any(np.isfinite(weighted), axis=0)
    if not np.all(any_finite):
        bad = int(np.argmin(any_finite))
        raise ZeroLikelihoodError(
            f"Site {bad} has zero likelihood under every rate category, so "
            "conditional substitution statistics are undefined at that site."
        )
    max_w = weighted.max(axis=0)
    site_lnl = max_w + np.log(np.exp(weighted - max_w[None, :]).sum(axis=0))
    rate_posteriors = np.exp(weighted - site_lnl[None, :])

    # Accumulators for per-branch conditional statistics.
    counts_sum = {id(node): np.zeros((n_sites, n_states, n_states)) for node in edges}
    dwell_sum = {id(node): np.zeros((n_sites, n_states)) for node in edges}

    Q_off = Q.copy()
    np.fill_diagonal(Q_off, 0.0)
    state_idx = np.arange(n_states)

    for k, rate in enumerate(rates):
        partials = down_partials[k]
        log_scales = down_log_scales[k]
        root_scale = log_scales[id(root)]
        root_dot = root_dots[k]
        valid_sites = root_dot > 0.0

        # Upward pass. The root "up" message is the root prior itself.
        up = {id(root): np.broadcast_to(pi, (n_sites, n_states)).copy()}
        up_scale = {id(root): np.zeros(n_sites)}

        for parent in preorder:
            if parent.is_terminal():
                continue
            base_up = up[id(parent)]
            base_scale = up_scale[id(parent)]
            for child in parent.clades:
                P_child = transitions[(k, id(child))]
                h = base_up.copy()
                sibling_scale = np.zeros(n_sites)
                for sibling in parent.clades:
                    if sibling is child:
                        continue
                    P_sib = transitions[(k, id(sibling))]
                    h = h * (partials[id(sibling)] @ P_sib.T)
                    sibling_scale = sibling_scale + log_scales[id(sibling)]
                h_sum = h.sum(axis=1)
                h_bar = np.zeros_like(h)
                ok = valid_sites & (h_sum > 0.0)
                h_bar[ok] = h[ok] / h_sum[ok, None]
                message = h_bar @ P_child
                msg_sum = message.sum(axis=1)
                up_child = np.zeros_like(message)
                msg_ok = ok & (msg_sum > 0.0)
                up_child[msg_ok] = message[msg_ok] / msg_sum[msg_ok, None]
                up[id(child)] = up_child
                child_up_scale = base_scale + sibling_scale
                child_up_scale[ok] = child_up_scale[ok] + np.log(h_sum[ok])
                child_up_scale[msg_ok] = child_up_scale[msg_ok] + np.log(
                    msg_sum[msg_ok]
                )
                up_scale[id(child)] = child_up_scale

                # Kernel contraction with the eigenbasis:
                #   u = h_bar V,  w = V^-1 d_child
                #   Z[s,a,l] = sum_m V^-1[m,a] u[s,m] Phi[m,l]
                #   R[s,a,b] = sum_l Z[s,a,l] w[s,l] V[b,l]
                phi = phis[(k, id(child))]
                u_vec = h_bar @ V_c
                w_vec = partials[id(child)] @ V_inv_c.T
                z_arr = np.einsum("ma,sm,ml->sal", V_inv_c, u_vec, phi, optimize=True)
                r_arr = np.einsum(
                    "sal,sl,bl->sab",
                    z_arr,
                    w_vec,
                    V_c,
                    optimize=True,
                )
                imag_tol = 1e-9 * max(1.0, np.abs(r_arr.real).max())
                if np.any(np.abs(r_arr.imag) > imag_tol):
                    raise FloatingPointError(
                        "Non-negligible imaginary part in the substitution "
                        "kernel; check the rate matrix."
                    )
                r_arr = np.asarray(r_arr.real, dtype=float)
                log_factor = base_scale + sibling_scale - root_scale
                log_factor[valid_sites] = log_factor[valid_sites] - np.log(
                    root_dot[valid_sites]
                )
                log_factor[ok] = log_factor[ok] + np.log(h_sum[ok])
                log_factor = log_factor + log_scales[id(child)]
                edge_factor = np.zeros(n_sites)
                edge_factor[ok] = np.exp(log_factor[ok])
                edge_factor[~valid_sites] = 0.0
                # Counts: reward B[i,j] = r Q[i,j]; dwell: B[i,i] = 1.
                counts_k = rate * Q_off[None, :, :] * r_arr
                dwell_k = r_arr[:, state_idx, state_idx]
                post = rate_posteriors[k]
                counts_contribution = (
                    post[:, None, None] * edge_factor[:, None, None] * counts_k
                )
                dwell_contribution = post[:, None] * edge_factor[:, None] * dwell_k
                counts_contribution[~valid_sites] = 0.0
                dwell_contribution[~valid_sites] = 0.0
                counts_sum[id(child)] = counts_sum[id(child)] + counts_contribution
                dwell_sum[id(child)] = dwell_sum[id(child)] + dwell_contribution

    # Build result records; clean up rounding-level negatives.
    branches = []
    max_rate = abs(rates).max()
    for node in edges:
        counts = np.asarray(counts_sum[id(node)], dtype=float)
        dwell = np.asarray(dwell_sum[id(node)], dtype=float)
        t = node.branch_length
        tol_counts = 1e-9 * max(1.0, t * max_rate)
        tol_dwell = 1e-9 * max(1.0, t)
        counts = np.where((counts < 0.0) & (counts > -tol_counts), 0.0, counts)
        dwell = np.where((dwell < 0.0) & (dwell > -tol_dwell), 0.0, dwell)
        counts[:, state_idx, state_idx] = 0.0
        branches.append(
            BranchSubstitutionStats(
                parent_of[id(node)],
                node,
                t,
                states,
                counts,
                dwell,
            )
        )

    return ConditionalSubstitutionStats(
        tree,
        states,
        rates,
        rate_weights,
        site_lnl,
        rate_posteriors.T,
        branches,
    )
