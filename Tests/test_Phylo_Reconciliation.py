# Copyright (C) 2026 by the Biopython Contributors
# This code is part of the Biopython distribution and governed by its
# license. Please see the LICENSE file that should have been included
# as part of this package.

"""Unit tests for the Bio.Phylo.Reconciliation module."""

import random
import unittest
from io import StringIO

from Bio import Phylo
from Bio.Phylo import BaseTree
from Bio.Phylo.Reconciliation import DUPLICATION
from Bio.Phylo.Reconciliation import SPECIATION
from Bio.Phylo.Reconciliation import TRANSFER
from Bio.Phylo.Reconciliation import reconcile_dtl


def _read(newick):
    """Read a rooted Newick tree from a string (PRIVATE)."""
    return Phylo.read(StringIO(newick), "newick", rooted=True)


def _clade(name=None, clades=None):
    """Build a BaseTree.Clade with the given children (PRIVATE)."""
    return BaseTree.Clade(name=name, clades=clades)


class ReconciliationExamplesTest(unittest.TestCase):
    """Hand-computed reconciliation scenarios."""

    def test_cospeciation_zero_cost(self):
        species = _read("((A:1,B:1):1,(C:1,D:1):1);")
        gene = _read("((a,b),(c,d));")
        result = reconcile_dtl(gene, species, {"a": "A", "b": "B", "c": "C", "d": "D"})
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(set(result.events.values()), {SPECIATION})
        self.assertEqual(result.num_losses, 0)
        self.assertEqual(result.num_duplications, 0)
        self.assertEqual(result.num_transfers, 0)
        for leaf in gene.get_terminals():
            self.assertEqual(result.gene_mapping[leaf].name, leaf.name.upper())

    def test_duplication_with_paralogs(self):
        # Two paralogs a1, a2 in species A plus an ortholog b in B.
        species = _read("(A:1,B:1);")
        gene = _read("((a1,a2),b);")
        result = reconcile_dtl(gene, species, {"a1": "A", "a2": "A", "b": "B"})
        self.assertEqual(result.num_duplications, 1)
        self.assertEqual(result.num_transfers, 0)
        self.assertEqual(result.num_losses, 0)
        self.assertEqual(result.total_cost, result.duplication_cost)
        paralog_node = gene.root.clades[0]
        self.assertEqual(result.events[paralog_node], DUPLICATION)
        self.assertEqual(result.gene_mapping[paralog_node].name, "A")

    def test_single_loss_explanation(self):
        # Gene (a, c) against ((A, B), C): lineage B is lost on the a edge.
        species = _read("((A:1,B:1):1,C:2);")
        gene = _read("(a,c);")
        result = reconcile_dtl(gene, species, {"a": "A", "c": "C"})
        self.assertEqual(result.total_cost, result.loss_cost)
        self.assertEqual(result.num_losses, 1)
        loss = result.losses[0]
        self.assertIs(loss.gene_parent, gene.root)
        self.assertEqual(loss.gene_child.name, "a")
        self.assertEqual(loss.lost_species.name, "B")
        self.assertIsNone(loss.species_node.name)  # the internal (A, B) node
        self.assertEqual(result.events[gene.root], SPECIATION)

    def test_transfer_preferred_when_cheap(self):
        # G=(a,d), S=((A,B),(C,D)): a transfer at the leaf time level beats
        # two losses once the transfer cost drops below 2 * loss cost.
        species = _read("((A:1,B:1):1,(C:1,D:1):1);")
        gene = _read("(a,d);")
        leaf_map = {"a": "A", "d": "D"}
        cheap = reconcile_dtl(gene, species, leaf_map, transfer_cost=1)
        self.assertEqual(cheap.num_transfers, 1)
        self.assertEqual(cheap.num_losses, 0)
        self.assertEqual(cheap.total_cost, 1)
        self.assertEqual(cheap.events[gene.root], TRANSFER)
        donor = cheap.gene_mapping[gene.root]
        recipient = cheap.transfer_targets[gene.root]
        transferred_child = next(
            child
            for child in gene.root.clades
            if cheap.gene_mapping[child] is recipient
        )
        # Transfer edges are instantaneous, at the shared leaf time level.
        self.assertEqual(
            cheap.event_times[gene.root],
            cheap.event_times[transferred_child],
        )
        self.assertEqual(cheap.event_times[gene.root], 0.0)
        # A transfer cannot land inside the donor's own species subtree.
        self.assertIsNot(recipient, donor)

    def test_losses_preferred_when_transfer_expensive(self):
        species = _read("((A:1,B:1):1,(C:1,D:1):1);")
        gene = _read("(a,d);")
        result = reconcile_dtl(
            gene, species, {"a": "A", "d": "D"}, transfer_cost=3, loss_cost=1
        )
        self.assertEqual(result.num_transfers, 0)
        self.assertEqual(result.num_losses, 2)
        self.assertEqual(result.total_cost, 2)
        lost_names = sorted(loss.lost_species.name for loss in result.losses)
        self.assertEqual(lost_names, ["B", "C"])

    def test_transfer_between_internal_time_levels(self):
        # G=((a,b),d), S=((A,B),(C,D)); with cheap transfer and costly loss
        # the (a,b) lineage transfers from CD to AB at the internal time level.
        species = _read("((A:1,B:1):1,(C:1,D:1):1);")
        gene = _read("((a,b),d);")
        result = reconcile_dtl(
            gene,
            species,
            {"a": "A", "b": "B", "d": "D"},
            transfer_cost=1,
            loss_cost=2,
        )
        self.assertEqual(result.num_transfers, 1)
        self.assertEqual(result.num_losses, 0)
        self.assertEqual(result.total_cost, 1)
        recipient = result.transfer_targets[gene.root]
        transferred_child = next(
            child
            for child in gene.root.clades
            if result.gene_mapping[child] is recipient
        )
        self.assertEqual(result.event_times[gene.root], 1.0)
        self.assertEqual(result.event_times[transferred_child], 1.0)
        self.assertEqual(sorted(child.name for child in recipient.clades), ["A", "B"])

    def test_multiple_genes_per_species(self):
        # Three gene leaves map to species A, one to B.
        species = _read("(A:1,B:1);")
        gene = _read("((a1,a2),(a3,b1));")
        result = reconcile_dtl(
            gene, species, {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
        )
        self.assertEqual(result.num_duplications, 2)
        self.assertEqual(result.num_transfers, 0)
        self.assertEqual(result.num_losses, 0)
        self.assertEqual(result.total_cost, 2 * result.duplication_cost)
        # All gene leaves keep their prescribed species assignment.
        for leaf in gene.get_terminals():
            expected = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}[leaf.name]
            self.assertEqual(result.gene_mapping[leaf].name, expected)

    def test_explicit_time_map_by_name(self):
        species = BaseTree.Tree(
            root=_clade(
                "root",
                [
                    _clade("AB", [_clade("A"), _clade("B")]),
                    _clade("C"),
                ],
            ),
            rooted=True,
        )
        gene = _read("(a,c);")
        time_map = {"root": 100, "AB": 10, "A": 0, "B": 0, "C": 0}
        result = reconcile_dtl(gene, species, {"a": "A", "c": "C"}, time_map=time_map)
        self.assertEqual(result.event_times[gene.root], 100)
        self.assertEqual(result.event_times[gene.root.clades[1]], 0)

    def test_time_attributes_on_nodes(self):
        species = _read("((A:1,B:1):1,C:2);")
        times = {
            node: value
            for node, value in zip(
                list(species.find_clades(order="preorder")), [9, 4, 0, 0, 0]
            )
        }
        for node, value in times.items():
            node.time = value
        gene = _read("(a,c);")
        result = reconcile_dtl(gene, species, {"a": "A", "c": "C"})
        self.assertEqual(result.event_times[gene.root], 9)

    def test_object_keyed_leaf_map(self):
        species = _read("(A:1,B:1);")
        gene = _read("(a,b);")
        species_by_name = {node.name: node for node in species.get_terminals()}
        leaf_map = {
            gene.get_terminals()[0]: species_by_name["A"],
            gene.get_terminals()[1]: species_by_name["B"],
        }
        result = reconcile_dtl(gene, species, leaf_map)
        self.assertEqual(result.total_cost, 0)
        self.assertIs(
            result.gene_mapping[gene.get_terminals()[0]], species_by_name["A"]
        )

    def test_single_gene_leaf(self):
        species = _read("(A:1,B:1);")
        gene = BaseTree.Tree(root=_clade("x"), rooted=True)
        result = reconcile_dtl(gene, species, {"x": "A"})
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.gene_mapping[gene.root].name, "A")
        self.assertEqual(result.events, {})
        self.assertEqual(result.losses, [])


class ReconciliationValidationTest(unittest.TestCase):
    """Input validation and error reporting."""

    def test_unrooted_tree_rejected(self):
        species = Phylo.read(StringIO("(A:1,B:1);"), "newick")
        gene = _read("(a,b);")
        self.assertFalse(species.rooted)
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "b": "B"})

    def test_polytomy_rejected(self):
        species = BaseTree.Tree(
            root=_clade(None, [_clade("A"), _clade("B"), _clade("C")]),
            rooted=True,
        )
        gene = _read("(a,b);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "b": "B"})

    def test_missing_leaf_mapping_rejected(self):
        species = _read("(A:1,B:1);")
        gene = _read("(a,b);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A"})

    def test_unknown_gene_leaf_name_rejected(self):
        species = _read("(A:1,B:1);")
        gene = _read("(a,b);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "z": "B"})

    def test_internal_species_mapping_rejected(self):
        species = _read("((A:1,B:1):1,C:2);")
        gene = _read("(a,c);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "c": "AB"})

    def test_non_positive_costs_rejected(self):
        species = _read("(A:1,B:1);")
        gene = _read("(a,b);")
        kwargs = {"a": "A", "b": "B"}
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, kwargs, duplication_cost=0)
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, kwargs, transfer_cost=-1)
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, kwargs, loss_cost=0)

    def test_non_decreasing_times_rejected(self):
        species = BaseTree.Tree(
            root=_clade("root", [_clade("A"), _clade("B")]), rooted=True
        )
        gene = _read("(a,b);")
        bad_times = {"root": 1, "A": 1, "B": 0}
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "b": "B"}, time_map=bad_times)

    def test_incomplete_time_map_rejected(self):
        species = BaseTree.Tree(
            root=_clade("root", [_clade("A"), _clade("B")]), rooted=True
        )
        gene = _read("(a,b);")
        with self.assertRaises(ValueError):
            reconcile_dtl(
                gene,
                species,
                {"a": "A", "b": "B"},
                time_map={"root": 1, "A": 0},
            )

    def test_non_ultrametric_branch_lengths_rejected(self):
        species = _read("((A:1,B:2):1,C:1);")
        gene = _read("(a,c);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "c": "C"})

    def test_duplicate_species_names_rejected(self):
        species = BaseTree.Tree(
            root=_clade(None, [_clade("A"), _clade("A")]), rooted=True
        )
        gene = _read("(a,b);")
        with self.assertRaises(ValueError):
            reconcile_dtl(gene, species, {"a": "A", "b": "A"})


class ReconciliationInvariantsTest(unittest.TestCase):
    """Structural invariants every returned reconciliation must satisfy."""

    def setUp(self):
        self.species = _read("((A:1,B:1):1,(C:1,D:1):1);")
        self.gene = _read("((a,b),(a2,d));")
        self.leaf_map = {"a": "A", "b": "B", "a2": "A", "d": "D"}
        self.result = reconcile_dtl(self.gene, self.species, self.leaf_map)

    def test_total_cost_matches_event_counts(self):
        expected = (
            self.result.duplication_cost * self.result.num_duplications
            + self.result.transfer_cost * self.result.num_transfers
            + self.result.loss_cost * self.result.num_losses
        )
        self.assertEqual(expected, self.result.total_cost)

    def test_every_gene_node_mapped_and_timed(self):
        gene_nodes = list(self.gene.find_clades())
        self.assertEqual(set(self.result.gene_mapping), set(gene_nodes))
        self.assertEqual(set(self.result.event_times), set(gene_nodes))
        internal = set(self.gene.get_nonterminals())
        self.assertEqual(set(self.result.events), internal)

    def test_leaf_mapping_preserved(self):
        for leaf in self.gene.get_terminals():
            self.assertEqual(
                self.result.gene_mapping[leaf].name, self.leaf_map[leaf.name]
            )

    def test_time_consistency(self):
        for gene_node in self.gene.get_nonterminals():
            parent_time = self.result.event_times[gene_node]
            recipient = self.result.transfer_targets.get(gene_node)
            for child in gene_node.clades:
                child_time = self.result.event_times[child]
                self.assertGreaterEqual(parent_time, child_time)
                if self.result.gene_mapping[child] is recipient:
                    # Transfer edges are instantaneous: equal event times.
                    self.assertEqual(parent_time, child_time)
                    self.assertEqual(self.result.events[gene_node], TRANSFER)

    def test_transfer_targets_contemporaneous_and_external(self):
        subtree_of = {
            node: set(node.find_clades()) for node in self.species.find_clades()
        }
        for gene_node, recipient in self.result.transfer_targets.items():
            donor = self.result.gene_mapping[gene_node]
            transferred_child = next(
                child
                for child in gene_node.clades
                if self.result.gene_mapping[child] is recipient
            )
            self.assertNotIn(recipient, subtree_of[donor])
            self.assertEqual(
                self.result.event_times[gene_node],
                self.result.event_times[transferred_child],
            )

    @staticmethod
    def _serialize(tree):
        handle = StringIO()
        Phylo.write([tree], handle, "newick")
        return handle.getvalue()

    def test_input_trees_not_modified(self):
        species_before = self._serialize(self.species)
        gene_before = self._serialize(self.gene)
        reconcile_dtl(self.gene, self.species, self.leaf_map)
        self.assertEqual(self._serialize(self.species), species_before)
        self.assertEqual(self._serialize(self.gene), gene_before)
        self.assertTrue(self.gene.rooted)
        self.assertTrue(self.species.rooted)


class BruteForceCrossCheckTest(unittest.TestCase):
    """Cross-check the O(mn) dynamic program against an independent O(mn^2) DP."""

    @staticmethod
    def _brute_force_cost(gene_tree, species_tree, leaf_map, d_cost, t_cost, l_cost):
        sp_post = list(species_tree.find_clades(order="postorder"))
        sp_pre = list(species_tree.find_clades(order="preorder"))
        parent = {}
        depth = {sp_pre[0]: 0}
        for node in sp_pre:
            for child in node.clades:
                depth[child] = depth[node] + 1
                parent[child] = node
        height = max(depth[n] for n in sp_post if n.is_terminal())
        time_of = {node: height - depth[node] for node in sp_post}
        subtree = {}
        for node in sp_post:
            descendants = (
                frozenset().union(*(subtree[child] for child in node.clades))
                if not node.is_terminal()
                else frozenset()
            )
            # The subtree of a node includes the node itself: child lineages
            # born at a speciation/duplication may stay on the same species.
            subtree[node] = descendants | frozenset([node])
        level = {}
        for node in sp_post:
            level.setdefault(time_of[node], set()).add(node)

        def edge_loss(species_node, target_node):
            distance = depth[target_node] - depth[species_node]
            return l_cost * max(0, distance - 1)

        costs = {}
        inf = float("inf")
        for gene_node in gene_tree.find_clades(order="postorder"):
            row = {node: inf for node in sp_post}
            if gene_node.is_terminal():
                target_name = leaf_map[gene_node.name]
                target = next(
                    node
                    for node in sp_post
                    if node.is_terminal() and node.name == target_name
                )
                row[target] = 0
            else:
                first, second = gene_node.clades
                for species_node in sp_post:
                    options = []
                    if not species_node.is_terminal():
                        left, right = species_node.clades
                        options.append(
                            min(
                                costs[first][x] + edge_loss(species_node, x)
                                for x in subtree[left]
                            )
                            + min(
                                costs[second][y] + edge_loss(species_node, y)
                                for y in subtree[right]
                            )
                        )
                        options.append(
                            min(
                                costs[second][x] + edge_loss(species_node, x)
                                for x in subtree[left]
                            )
                            + min(
                                costs[first][y] + edge_loss(species_node, y)
                                for y in subtree[right]
                            )
                        )
                    options.append(
                        d_cost
                        + min(
                            costs[first][x] + edge_loss(species_node, x)
                            for x in subtree[species_node]
                        )
                        + min(
                            costs[second][y] + edge_loss(species_node, y)
                            for y in subtree[species_node]
                        )
                    )
                    for retained, transferred in ((first, second), (second, first)):
                        landings = [
                            costs[transferred][node]
                            for node in level[time_of[species_node]]
                            if node is not species_node
                        ]
                        if landings:
                            options.append(
                                t_cost
                                + min(
                                    costs[retained][x] + edge_loss(species_node, x)
                                    for x in subtree[species_node]
                                )
                                + min(landings)
                            )
                    row[species_node] = min(options)
            costs[gene_node] = row
        return min(costs[gene_tree.root].values())

    def test_random_trees(self):
        rng = random.Random(1729)
        cost_settings = [
            (2, 3, 1),
            (1, 1, 1),
            (5, 2, 1),
            (2, 10, 1),
            (3, 3, 3),
            (10, 1, 10),
        ]
        for seed in range(40):
            # Tree.randomized draws from the global random module.
            random.seed(seed)
            rng.seed(seed)
            species_size = rng.randint(2, 6)
            gene_size = rng.randint(2, 8)
            species = BaseTree.Tree.randomized(species_size, branch_length=1.0)
            species.rooted = True
            gene = BaseTree.Tree.randomized(gene_size, branch_length=1.0)
            gene.rooted = True
            # Random Yule trees are not ultrametric, so supply explicit
            # node times from topological depth, as in the brute force DP.
            depth = species.depths(unit_branch_lengths=True)
            height = max(depth[node] for node in species.get_terminals())
            time_map = {node: height - depth[node] for node in species.find_clades()}
            species_names = [node.name for node in species.get_terminals()]
            leaf_map = {
                node.name: rng.choice(species_names) for node in gene.get_terminals()
            }
            d_cost, t_cost, l_cost = cost_settings[seed % len(cost_settings)]
            result = reconcile_dtl(
                gene,
                species,
                leaf_map,
                duplication_cost=d_cost,
                transfer_cost=t_cost,
                loss_cost=l_cost,
                time_map=time_map,
            )
            expected = self._brute_force_cost(
                gene, species, leaf_map, d_cost, t_cost, l_cost
            )
            self.assertEqual(
                result.total_cost,
                expected,
                msg=(
                    f"seed {seed}, sizes {species_size}/{gene_size}, "
                    f"costs {(d_cost, t_cost, l_cost)}"
                ),
            )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
