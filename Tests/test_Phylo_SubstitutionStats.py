# Copyright (C) 2026 by The Biopython Contributors
# This code is part of the Biopython distribution and governed by its
# license. Please see the LICENSE file that should have been included
# as part of this package.

"""Unit tests for Bio.Phylo.SubstitutionStats (conditional mapping)."""

import unittest
from io import StringIO

import numpy as np

from Bio import Phylo
from Bio.Align import Alignment
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo import BaseTree
from Bio.Phylo.SubstitutionStats import BranchSubstitutionStats
from Bio.Phylo.SubstitutionStats import ConditionalSubstitutionStats
from Bio.Phylo.SubstitutionStats import ZeroLikelihoodError
from Bio.Phylo.SubstitutionStats import _build_character_vectors
from Bio.Phylo.SubstitutionStats import conditional_substitution_stats
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# --------------------------------------------------------------------------
# Independent reference implementation: plain pruning/upward messages with
# 2n x 2n block-matrix exponentials (scaling and squaring, Taylor series).
# It shares no linear algebra with the production code (no eigen kernel).
# --------------------------------------------------------------------------


def _expm(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norm = np.linalg.norm(matrix, ord=np.inf)
    squarings = max(0, int(np.ceil(np.log2(norm)))) if norm > 0 else 0
    x = matrix / 2**squarings
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for k in range(1, 100):
        term = term @ x / k
        result = result + term
        if np.abs(term).max() < 1e-20:
            break
    for _ in range(squarings):
        result = result @ result
    return result


def _integral_kernel(rate, t, Q, reward):
    n = Q.shape[0]
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = rate * Q
    block[n:, n:] = rate * Q
    block[:n, n:] = reward
    return _expm(t * block)[:n, n:]


def _reference_stats(tree, rows, Q, pi, rate_mixture, states, character_vectors=None):
    """Compute conditional stats independently; return dict by child id."""
    Q = np.asarray(Q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    n = Q.shape[0]
    states = tuple(states)
    n_sites = len(next(iter(rows.values())))
    cmap = _build_character_vectors(states, character_vectors)

    root = tree.root
    preorder = []
    stack = [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(reversed(node.clades))
    postorder = list(reversed(preorder))
    terminals = [node for node in preorder if not node.clades]

    obs = {}
    for node in terminals:
        obs[node] = np.vstack([cmap[ch] for ch in rows[node.name]])

    rates = np.array([pair[0] for pair in rate_mixture])
    weights = np.array([pair[1] for pair in rate_mixture])
    weights = weights / weights.sum()

    expected = {}
    likelihoods = np.empty((len(rates), n_sites))
    per_category = {}

    for k, rate in enumerate(rates):
        down = {}
        transition = {}
        for node in postorder:
            if node.is_terminal():
                down[node] = obs[node].copy()
                transition[node] = None
                continue
            g = np.ones((n_sites, n))
            for child in node.clades:
                t = child.branch_length
                if transition[child] is None:
                    transition[child] = _expm(rate * t * Q)
                g = g * (down[child] @ transition[child].T)
            down[node] = g
            transition[node] = None
        likelihood = down[root] @ pi
        likelihoods[k] = likelihood

        up = {root: np.broadcast_to(pi, (n_sites, n)).copy()}
        per_category[k] = {}
        for parent in preorder:
            if parent.is_terminal():
                continue
            for child in parent.clades:
                if transition[child] is None:
                    transition[child] = _expm(rate * child.branch_length * Q)
                f_arr = up[parent].copy()
                for sibling in parent.clades:
                    if sibling is child:
                        continue
                    if transition[sibling] is None:
                        transition[sibling] = _expm(rate * sibling.branch_length * Q)
                    f_arr = f_arr * (down[sibling] @ transition[sibling].T)
                up[child] = f_arr @ transition[child]

                t = child.branch_length
                counts = np.zeros((n_sites, n, n))
                dwell = np.zeros((n_sites, n))
                valid = likelihood > 0.0
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        reward = np.zeros((n, n))
                        reward[a, b] = rate * Q[a, b]
                        kernel = _integral_kernel(rate, t, Q, reward)
                        numerator = np.einsum("si,ij,sj->s", f_arr, kernel, down[child])
                        counts[valid, a, b] = numerator[valid] / likelihood[valid]
                    reward = np.zeros((n, n))
                    reward[a, a] = 1.0
                    kernel = _integral_kernel(rate, t, Q, reward)
                    numerator = np.einsum("si,ij,sj->s", f_arr, kernel, down[child])
                    dwell[valid, a] = numerator[valid] / likelihood[valid]
                per_category[k][id(child)] = (counts, dwell)

    # Mix categories using posterior weights.
    with np.errstate(divide="ignore"):
        log_w = np.log(weights)[:, None] + np.log(likelihoods)
    max_w = log_w.max(axis=0)
    posterior = np.exp(
        log_w
        - max_w[None, :]
        - np.log(np.exp(log_w - max_w[None, :]).sum(axis=0))[None, :]
    )
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    for child in preorder:
        if child is root:
            continue
        counts = sum(
            posterior[k][:, None, None] * per_category[k][id(child)][0]
            for k in range(len(rates))
        )
        dwell = sum(
            posterior[k][:, None] * per_category[k][id(child)][1]
            for k in range(len(rates))
        )
        expected[id(child)] = (counts, dwell)
    return expected


def _make_alignment(sequences):
    return MultipleSeqAlignment(
        [SeqRecord(Seq(seq), id=name) for name, seq in sequences.items()]
    )


def _binary_tree(length_a=0.7, length_b=0.3):
    root = BaseTree.Clade()
    leaf_a = BaseTree.Clade(branch_length=length_a, name="A")
    leaf_b = BaseTree.Clade(branch_length=length_b, name="B")
    root.clades = [leaf_a, leaf_b]
    return BaseTree.Tree(root, rooted=True), root, leaf_a, leaf_b


def _hky_matrix(pi, kappa=4.0):
    pi = np.asarray(pi, dtype=float)
    n = 4
    Q = np.zeros((n, n))
    transitions = {(0, 2), (2, 0), (1, 3), (3, 1)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Q[i, j] = (kappa if (i, j) in transitions else 1.0) * pi[j]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    scale = -pi @ np.diag(Q)
    return Q / scale


class InvariantTests(unittest.TestCase):
    """Structural identities valid for any site/branch."""

    def setUp(self):
        self.tree, self.root, self.leaf_a, self.leaf_b = _binary_tree()
        self.alignment = _make_alignment({"A": "ACN-", "B": "A-AN"})
        self.Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        self.result = conditional_substitution_stats(
            self.tree,
            self.alignment,
            self.Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )

    def test_container(self):
        self.assertIsInstance(self.result, ConditionalSubstitutionStats)
        self.assertEqual(len(self.result), 2)
        self.assertEqual({b.clade.name for b in self.result}, {"A", "B"})
        self.assertIs(self.result[self.leaf_a].clade, self.leaf_a)
        self.assertIs(self.result["B"].clade, self.leaf_b)
        self.assertIn("A", self.result)
        self.assertIn(self.leaf_b, self.result)
        self.assertIsNone(self.result.get("missing"))
        with self.assertRaises(KeyError):
            self.result["missing"]

    def test_shapes_and_diagonal(self):
        for branch in self.result:
            self.assertIsInstance(branch, BranchSubstitutionStats)
            self.assertEqual(branch.counts.shape, (4, 2, 2))
            self.assertEqual(branch.dwell.shape, (4, 2))
            self.assertTrue(np.allclose(branch.counts.diagonal(axis1=1, axis2=2), 0.0))
            self.assertTrue(np.all(branch.counts >= -1e-12))
            self.assertTrue(np.all(branch.dwell >= -1e-12))

    def test_dwell_sums_to_branch_length(self):
        self.assertTrue(np.allclose(self.result["A"].dwell.sum(axis=1), 0.7))
        self.assertTrue(np.allclose(self.result["B"].dwell.sum(axis=1), 0.3))

    def test_posteriors_and_likelihood(self):
        self.assertTrue(np.allclose(self.result.rate_posteriors.sum(axis=1), 1.0))
        self.assertEqual(self.result.rate_posteriors.shape, (4, 1))
        self.assertTrue(
            np.isclose(
                self.result.log_likelihood,
                self.result.site_log_likelihoods.sum(),
            )
        )

    def test_hidden_back_and_forth_counted(self):
        # Identical endpoints on a long branch still accumulate expected
        # substitutions (a parsimony map would report zero changes).
        tree, root, leaf_a, leaf_b = _binary_tree(5.0, 5.0)
        alignment = _make_alignment({"A": "A", "B": "A"})
        result = conditional_substitution_stats(
            tree,
            alignment,
            self.Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )
        for branch in result:
            self.assertGreater(branch.counts[0, 0, 1], 1.0)
            self.assertGreater(branch.counts[0, 1, 0], 1.0)

    def test_zero_length_branches(self):
        tree, root, leaf_a, leaf_b = _binary_tree(0.0, 0.0)
        alignment = _make_alignment({"A": "AC", "B": "AC"})
        result = conditional_substitution_stats(
            tree,
            alignment,
            self.Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )
        for branch in result:
            self.assertTrue(np.allclose(branch.counts, 0.0))
            self.assertTrue(np.allclose(branch.dwell, 0.0))


class ReferenceComparisonTests(unittest.TestCase):
    """Cross-check against the independent block-exponential reference."""

    def _assert_matches_reference(
        self,
        tree,
        rows,
        Q,
        pi,
        mixture,
        states,
        character_vectors=None,
        tol=1e-8,
    ):
        alignment = _make_alignment(rows)
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            root_distribution=pi,
            rate_mixture=mixture,
            states=states,
            character_vectors=character_vectors,
        )
        expected = _reference_stats(
            tree, rows, Q, pi, mixture, states, character_vectors
        )
        for branch in result:
            ref_counts, ref_dwell = expected[id(branch.clade)]
            self.assertTrue(
                np.allclose(branch.counts, ref_counts, atol=tol, rtol=tol),
                msg=f"counts mismatch on {branch.clade.name}",
            )
            self.assertTrue(
                np.allclose(branch.dwell, ref_dwell, atol=tol, rtol=tol),
                msg=f"dwell mismatch on {branch.clade.name}",
            )

    def test_reversible_four_state_with_mixture(self):
        tree = Phylo.read(
            StringIO("(A:0.12,B:0.37,(C:0.22,D:0.9,E:0.05)F:0.31);"),
            "newick",
        )
        tree.rooted = True
        pi = [0.1, 0.2, 0.3, 0.4]
        Q = _hky_matrix(pi)
        rows = {
            "A": "AGR",
            "B": "GT-",
            "C": "ANA",
            "D": "CGG",
            "E": "TAC",
        }
        self._assert_matches_reference(
            tree,
            rows,
            Q,
            pi,
            [(0.3, 0.6), (2.0, 0.4)],
            "ACGT",
        )

    def test_nonreversible_three_state(self):
        tree = Phylo.read(
            StringIO("(X:0.11,Y:0.64,(Z:0.42,W:0.9)U:0.28);"),
            "newick",
        )
        tree.rooted = True
        Q = np.array(
            [
                [-0.2, 0.15, 0.05],
                [0.1, -0.2, 0.1],
                [0.2, 0.2, -0.4],
            ]
        )
        pi = [0.25, 0.35, 0.4]
        rows = {"X": "01", "Y": "2-", "Z": "02", "W": "11"}
        self._assert_matches_reference(tree, rows, Q, pi, [(1.0, 1.0)], "012")

    def test_binary_tree(self):
        tree, _, _, _ = _binary_tree()
        rows = {"A": "ACN-", "B": "A-AN"}
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        self._assert_matches_reference(tree, rows, Q, [0.5, 0.5], [(1.0, 1.0)], "AC")

    def test_polytomy(self):
        tree = Phylo.read(StringIO("(A:0.44,B:0.09,C:1.3,D:0.7);"), "newick")
        tree.rooted = True
        Q = np.array([[-3.0, 1.0, 2.0], [1.0, -2.0, 1.0], [2.0, 1.0, -3.0]])
        pi = np.array([0.3, 0.3, 0.4])
        pi = pi / pi.sum()
        rows = {"A": "01-", "B": "220", "C": "0-1", "D": "102"}
        self._assert_matches_reference(
            tree, rows, Q, pi, [(0.5, 0.25), (1.5, 0.75)], "012"
        )

    def test_rate_zero_category(self):
        tree, _, _, _ = _binary_tree(0.3, 0.3)
        rows = {"A": "A", "B": "C"}
        pi = [0.25, 0.25, 0.25, 0.25]
        Q = _hky_matrix(pi)
        self._assert_matches_reference(
            tree, rows, Q, pi, [(0.0, 0.3), (1.0, 0.7)], "ACGT"
        )

    def test_random_ten_tip_binary_tree(self):
        rng = np.random.default_rng(42)
        names = [f"S{i}" for i in range(10)]

        def random_newick(active):
            active = list(active)
            while len(active) > 1:
                i, j = rng.choice(len(active), 2, replace=False)
                t1 = 0.05 + rng.random() * 0.8
                t2 = 0.05 + rng.random() * 0.8
                a, b = active[i], active[j]
                active = [x for k, x in enumerate(active) if k not in (i, j)]
                active.append(f"({a}:{t1:.3f},{b}:{t2:.3f})")
            return active[0] + ";"

        tree = Phylo.read(StringIO(random_newick(names)), "newick")
        tree.rooted = True
        pi = np.array([0.1, 0.2, 0.3, 0.4])
        Q = _hky_matrix(pi)
        alphabet = "ACGT-"
        rows = {
            name: "".join(alphabet[i] for i in rng.integers(0, 5, size=6))
            for name in names
        }
        self._assert_matches_reference(
            tree,
            rows,
            Q,
            pi,
            [(0.25, 0.2), (1.0, 0.5), (3.0, 0.3)],
            "ACGT",
            tol=1e-8,
        )


class StationaryExpectationTests(unittest.TestCase):
    """Analytic expectations when the alignment is fully uninformative."""

    def test_binary_stationary(self):
        tree = Phylo.read(StringIO("(A:0.4,B:0.9,C:0.2);"), "newick")
        tree.rooted = True
        alignment = _make_alignment({"A": "---", "B": "NNN", "C": "???"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )
        for branch in result:
            t = branch.branch_length
            self.assertTrue(np.allclose(branch.dwell, 0.5 * t))
            self.assertTrue(np.allclose(branch.counts[:, 0, 1], 0.5 * t))
            self.assertTrue(np.allclose(branch.counts[:, 1, 0], 0.5 * t))
            self.assertTrue(np.allclose(branch.counts[:, range(2), range(2)], 0))

    def test_protein_stationary(self):
        from Bio.Data.IUPACData import protein_letters

        n = 20
        tree = Phylo.read(StringIO("(A:0.3,B:0.3,C:0.3);"), "newick")
        tree.rooted = True
        alignment = _make_alignment({"A": "X", "B": "-", "C": "?"})
        Q = np.full((n, n), 1.0 / (n - 1))
        np.fill_diagonal(Q, -1.0)
        result = conditional_substitution_stats(
            tree, alignment, Q, states=protein_letters
        )
        for branch in result:
            t = branch.branch_length
            self.assertTrue(np.allclose(branch.dwell, t / n, atol=1e-10))
            self.assertTrue(
                np.allclose(branch.counts[:, 0, 1], t / n / (n - 1), atol=1e-10)
            )
            self.assertAlmostEqual(branch.counts[0].sum(), t, places=10)


class AmbiguityTests(unittest.TestCase):
    def test_partial_ambiguity_is_likelihood_average(self):
        tree, _, _, _ = _binary_tree(0.4, 0.8)
        ambiguous = _make_alignment({"A": "R", "B": "G"})
        resolved = _make_alignment({"A": "AG", "B": "GG"})
        pi = [0.1, 0.2, 0.3, 0.4]
        Q = _hky_matrix(pi)
        result_amb = conditional_substitution_stats(
            tree, ambiguous, Q, root_distribution=pi, states="ACGT"
        )
        result_res = conditional_substitution_stats(
            tree, resolved, Q, root_distribution=pi, states="ACGT"
        )
        lnl = result_res.site_log_likelihoods
        weights = np.exp(lnl - np.log(np.exp(lnl).sum()))
        for branch_amb, branch_res in zip(result_amb, result_res):
            counts = (
                weights[0] * branch_res.counts[0] + weights[1] * branch_res.counts[1]
            )
            dwell = weights[0] * branch_res.dwell[0] + weights[1] * branch_res.dwell[1]
            self.assertTrue(np.allclose(branch_amb.counts[0], counts))
            self.assertTrue(np.allclose(branch_amb.dwell[0], dwell))

    def test_builtin_dna_codes_present(self):
        cmap = _build_character_vectors(tuple("ACGT"), None)
        self.assertTrue(np.array_equal(cmap["R"], [1, 0, 1, 0]))
        self.assertTrue(np.array_equal(cmap["Y"], [0, 1, 0, 1]))
        self.assertTrue(np.array_equal(cmap["N"], np.ones(4)))
        self.assertTrue(np.array_equal(cmap["-"], np.ones(4)))

    def test_custom_character_vectors(self):
        # State labels as iterable and as explicit weight vector.
        cmap = _build_character_vectors(tuple("012"), {"x": "02", "y": [0.0, 1.0, 2.0]})
        self.assertTrue(np.array_equal(cmap["x"], [1, 0, 1]))
        self.assertTrue(np.array_equal(cmap["y"], [0, 1, 2]))
        tree, _, _, _ = _binary_tree()
        alignment = _make_alignment({"A": "x0", "B": "y1"})
        Q = np.array([[-1.0, 1.0, 0.0], [0.5, -1.0, 0.5], [0.0, 1.0, -1.0]])
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="012",
            root_distribution=[1 / 3] * 3,
            character_vectors={"x": "02", "y": [0.0, 1.0, 2.0]},
        )
        self.assertTrue(np.all(np.isfinite(result["A"].counts)))


class RateMixtureTests(unittest.TestCase):
    def test_mixture_matches_posterior_weighted_single_runs(self):
        tree, _, _, _ = _binary_tree(0.35, 0.85)
        alignment = _make_alignment({"A": "ACA", "B": "A-C"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        mixture = [(0.4, 0.3), (1.6, 0.7)]
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[0.5, 0.5],
            rate_mixture=mixture,
        )
        single = []
        for rate, weight in mixture:
            single.append(
                (
                    weight,
                    conditional_substitution_stats(
                        tree,
                        alignment,
                        Q,
                        states="AC",
                        root_distribution=[0.5, 0.5],
                        rate_mixture=[(rate, 1.0)],
                    ),
                )
            )
        lnl_mix = result.site_log_likelihoods
        for branch in result:
            counts = np.zeros_like(branch.counts)
            dwell = np.zeros_like(branch.dwell)
            for weight, res in single:
                single_branch = res[branch.clade]
                post = weight * np.exp(res.site_log_likelihoods - lnl_mix)
                counts += post[:, None, None] * single_branch.counts
                dwell += post[:, None] * single_branch.dwell
            self.assertTrue(np.allclose(branch.counts, counts))
            self.assertTrue(np.allclose(branch.dwell, dwell))

    def test_equal_weights_short_form(self):
        tree, _, _, _ = _binary_tree()
        alignment = _make_alignment({"A": "A", "B": "C"})
        pi = [0.25, 0.25, 0.25, 0.25]
        Q = _hky_matrix(pi)
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            root_distribution=pi,
            states="ACGT",
            rate_mixture=[0.5, 2.0],
        )
        self.assertTrue(np.allclose(result.rate_weights, [0.5, 0.5]))
        self.assertTrue(np.allclose(result.rate_posteriors.sum(axis=1), 1.0))


class LikelihoodEdgeTests(unittest.TestCase):
    def test_zero_likelihood_raises(self):
        # Zero-length branches ending in a state the root cannot occupy:
        # likelihood is exactly zero, so the mapping is undefined.
        tree, _, _, _ = _binary_tree(0.0, 0.0)
        alignment = _make_alignment({"A": "C", "B": "C"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        with self.assertRaises(ZeroLikelihoodError) as ctx:
            conditional_substitution_stats(
                tree,
                alignment,
                Q,
                states="AC",
                root_distribution=[1.0, 0.0],
            )
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIn("0", str(ctx.exception))

    def test_zero_likelihood_later_site_index(self):
        tree, _, _, _ = _binary_tree(0.0, 0.0)
        alignment = _make_alignment({"A": "AC", "B": "AC"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        with self.assertRaises(ZeroLikelihoodError) as ctx:
            conditional_substitution_stats(
                tree,
                alignment,
                Q,
                states="AC",
                root_distribution=[1.0, 0.0],
            )
        self.assertIn("1", str(ctx.exception))

    def test_tiny_likelihood_stable(self):
        epsilon = 1e-80
        tree, _, _, _ = _binary_tree(0.0, 0.0)
        alignment = _make_alignment({"A": "C", "B": "C"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[1.0 - epsilon, epsilon],
        )
        self.assertTrue(np.all(np.isfinite(result["A"].counts)))
        self.assertTrue(np.all(np.isfinite(result["A"].dwell)))
        # At zero branch length both leaves repeat the root state, so the
        # site likelihood is pi_C = epsilon (one root draw, not two).
        self.assertAlmostEqual(
            result.site_log_likelihoods[0], np.log(epsilon), places=10
        )

    def test_stationary_root_distribution_default(self):
        tree, _, _, _ = _binary_tree()
        alignment = _make_alignment({"A": "AC", "B": "AA"})
        Q = np.array([[-2.0, 2.0], [3.0, -3.0]])
        explicit = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[0.6, 0.4],
        )
        default = conditional_substitution_stats(tree, alignment, Q, states="AC")
        for b1, b2 in zip(explicit, default):
            self.assertTrue(np.allclose(b1.counts, b2.counts))
            self.assertTrue(np.allclose(b1.dwell, b2.dwell))


class TreeAndInputTests(unittest.TestCase):
    def test_newick_roundtrip_unchanged(self):
        newick = "(A:0.12,B:0.37,(C:0.22,D:0.9)E:0.31);"
        tree = Phylo.read(StringIO(newick), "newick")
        tree.rooted = True
        before = tree.format("newick")
        pi = [0.25] * 4
        Q = _hky_matrix(pi)
        alignment = _make_alignment({"A": "AC", "B": "GT", "C": "AA", "D": "CG"})
        result = conditional_substitution_stats(
            tree, alignment, Q, root_distribution=pi, states="ACGT"
        )
        after = tree.format("newick")
        self.assertEqual(before, after)
        self.assertIs(result.tree, tree)

    def test_bio_align_alignment_object(self):
        tree, _, _, _ = _binary_tree()
        sequences = [
            SeqRecord(Seq("AC"), id="A"),
            SeqRecord(Seq("A-"), id="B"),
        ]
        coordinates = np.array([[0, 2], [0, 2]])
        alignment = Alignment(sequences, coordinates)
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(np.allclose(result["A"].dwell.sum(axis=1), 0.7))

    def test_validation_errors(self):
        good_tree, _, _, _ = _binary_tree()
        good_aln = _make_alignment({"A": "A", "B": "A"})
        good_Q = np.array([[-1.0, 1.0], [1.0, -1.0]])

        unrooted = BaseTree.Tree(good_tree.root, rooted=False)
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                unrooted,
                good_aln,
                good_Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )

        tree = BaseTree.Tree(
            BaseTree.Clade(
                clades=[
                    BaseTree.Clade(branch_length=0.1, name="A"),
                    BaseTree.Clade(branch_length=None, name="B"),
                ]
            ),
            rooted=True,
        )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                tree,
                good_aln,
                good_Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )

        tree_neg, _, _, _ = _binary_tree(-0.1, 0.2)
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                tree_neg,
                good_aln,
                good_Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )

        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                [[1.0]],
                states="AC",
                root_distribution=[0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                [[0.0, -1.0], [1.0, -1.0]],
                states="AC",
                root_distribution=[0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                [[-0.5, 0.1], [1.0, -1.0]],
                states="AC",
                root_distribution=[0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                good_Q,
                states="AC",
                root_distribution=[0.5, 0.6],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                good_Q,
                states="A",
                root_distribution=[0.5],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                good_tree,
                good_aln,
                good_Q,
                states="AA",
                root_distribution=[0.5, 0.5],
            )

    def test_alignment_taxon_mismatch(self):
        tree, _, _, _ = _binary_tree()
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                tree,
                _make_alignment({"A": "A", "X": "A"}),
                Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                tree,
                _make_alignment({"A": "A", "B": "A", "C": "A"}),
                Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )

    def test_unknown_character(self):
        tree, _, _, _ = _binary_tree()
        alignment = _make_alignment({"A": "G", "B": "A"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        with self.assertRaises(ValueError):
            conditional_substitution_stats(
                tree,
                alignment,
                Q,
                states="AC",
                root_distribution=[0.5, 0.5],
            )


class AggregationTests(unittest.TestCase):
    def test_totals_and_summing(self):
        tree, _, _, _ = _binary_tree(0.4, 0.6)
        alignment = _make_alignment({"A": "ACA", "B": "A-C"})
        Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
        result = conditional_substitution_stats(
            tree,
            alignment,
            Q,
            states="AC",
            root_distribution=[0.5, 0.5],
        )
        counts = result.total_counts()
        dwell = result.total_dwell()
        self.assertTrue(
            np.allclose(
                counts,
                sum(b.counts.sum(axis=0) for b in result),
            )
        )
        self.assertTrue(
            np.allclose(
                dwell,
                sum(b.dwell.sum(axis=0) for b in result),
            )
        )
        subset = result.total_counts(sites=[0, 2])
        self.assertTrue(
            np.allclose(
                subset,
                sum(b.counts[[0, 2]].sum(axis=0) for b in result),
            )
        )
        branch = result["A"]
        c_all, d_all = branch.sum()
        self.assertTrue(np.allclose(c_all, branch.counts.sum(axis=0)))
        self.assertTrue(np.allclose(d_all, branch.dwell.sum(axis=0)))
        c0, d0 = branch.sum(0)
        self.assertTrue(np.allclose(c0, branch.counts[0]))
        self.assertTrue(np.allclose(d0, branch.dwell[0]))
        # Total dwell over the tree equals total branch length times the
        # number of sites.
        self.assertAlmostEqual(dwell.sum(), 3.0, places=10)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
