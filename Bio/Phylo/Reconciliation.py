# Copyright (C) 2026 by the Biopython Contributors
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Native gene-tree/species-tree reconciliation on dated species trees.

This module implements a most-parsimonious Duplication-Transfer-Loss (DTL)
reconciliation for a binary rooted gene tree against a binary rooted, fully
dated species tree.  It is a pure-Python implementation of the dynamic
programming algorithm of Bansal, Alm and Kellis (2012), *Efficient algorithms
for the reconciliation problem with gene duplication, horizontal transfer and
loss*, Bioinformatics 28:i283-i291, and therefore requires no external
programs.

Because every node of the species tree carries a time stamp and time strictly
decreases along every root-to-leaf path, a horizontal transfer is only allowed
between species nodes that share the same time stamp.  This makes every
reconciliation returned here globally time-consistent: the event time of a
gene node is always greater than or equal to the event times of its children,
with equality only on a transfer edge.

The input trees are never modified, so all existing ``Bio.Phylo`` tree
operations keep working unchanged.

Example
-------
>>> from io import StringIO
>>> from Bio import Phylo
>>> from Bio.Phylo.Reconciliation import reconcile_dtl
>>> species = Phylo.read(
...     StringIO("((A:1,B:1):1,C:2);"), "newick", rooted=True)
>>> gene = Phylo.read(StringIO("(a,c);"), "newick", rooted=True)
>>> result = reconcile_dtl(
...     gene, species, {"a": "A", "c": "C"},
...     duplication_cost=2, transfer_cost=3, loss_cost=1)
>>> result.total_cost
1
>>> result.losses[0].lost_species.name
'B'
>>> result.event_times[gene.root]
2.0
"""

from collections import namedtuple

from Bio.Phylo import BaseTree

# Event type labels used in ``DTLReconciliation.events``.
SPECIATION = "speciation"
DUPLICATION = "duplication"
TRANSFER = "transfer"

# Internal cost sentinel: larger than any achievable reconciliation cost.
_INF = 10**18

# Choice codes stored in the dynamic programming table.
_CHOICE_S = 0
_CHOICE_D = 1
_CHOICE_T = 2


class Loss(
    namedtuple("Loss", ["gene_parent", "gene_child", "species_node", "lost_species"])
):
    """A single gene loss explanation.

    :param gene_parent: gene node on the parent side of the gene-tree edge
        carrying the loss.
    :param gene_child: gene node on the child side of the same gene-tree edge.
    :param species_node: species-tree node traversed by the gene lineage, where
        one daughter clade is followed and the other is lost.
    :param lost_species: root clade of the species-tree branch that retains no
        copy of the gene.
    """


class DTLReconciliation:
    """Result of a most-parsimonious DTL reconciliation.

    Attributes are read-only descriptions of the reconciliation; the input
    gene and species trees are referenced but never modified.

    :ivar gene_tree: the reconciled gene tree.
    :ivar species_tree: the dated species tree.
    :ivar gene_mapping: dict mapping every gene-tree node (including leaves)
        to the species-tree node on which it resides.
    :ivar events: dict mapping every internal gene-tree node to one of
        ``SPECIATION``, ``DUPLICATION`` or ``TRANSFER``.
    :ivar event_times: dict mapping every gene-tree node to the time of its
        mapped species-tree node, i.e. the time at which its event (or its
        presence at a leaf) occurs.
    :ivar transfer_targets: dict mapping each transferring gene-tree node to
        the recipient species-tree node; the donor species is the node in
        ``gene_mapping``.
    :ivar losses: list of :class:`Loss` records explaining every gene loss.
    :ivar total_cost: minimum total reconciliation cost.
    :ivar duplication_cost: duplication cost used.
    :ivar transfer_cost: transfer cost used.
    :ivar loss_cost: loss cost used.
    :ivar num_duplications: number of duplication events.
    :ivar num_transfers: number of transfer events.
    :ivar num_losses: number of loss events.
    """

    def __init__(
        self,
        gene_tree,
        species_tree,
        gene_mapping,
        events,
        event_times,
        transfer_targets,
        losses,
        total_cost,
        duplication_cost,
        transfer_cost,
        loss_cost,
    ):
        """Initialize a reconciliation result (PRIVATE)."""
        self.gene_tree = gene_tree
        self.species_tree = species_tree
        self.gene_mapping = gene_mapping
        self.events = events
        self.event_times = event_times
        self.transfer_targets = transfer_targets
        self.losses = losses
        self.total_cost = total_cost
        self.duplication_cost = duplication_cost
        self.transfer_cost = transfer_cost
        self.loss_cost = loss_cost
        self.num_duplications = sum(
            1 for event in events.values() if event == DUPLICATION
        )
        self.num_transfers = sum(1 for event in events.values() if event == TRANSFER)
        self.num_losses = len(losses)

    def __repr__(self):
        """Return a concise representation of the reconciliation."""
        return (
            "DTLReconciliation("
            f"duplications={self.num_duplications}, "
            f"transfers={self.num_transfers}, losses={self.num_losses}, "
            f"total_cost={self.total_cost})"
        )

    @staticmethod
    def _label(node):
        """Return a readable label for a tree node (PRIVATE)."""
        if node.name is not None:
            return str(node.name)
        return node.__class__.__name__

    def summary(self):
        """Return a human-readable, line-oriented description of the result.

        Gene-tree nodes are listed in preorder, each with its mapped species,
        event time and event type (for internal nodes).
        """
        lines = [
            "DTL reconciliation: "
            f"{self.num_duplications} duplication(s), "
            f"{self.num_transfers} transfer(s), "
            f"{self.num_losses} loss(es); total cost {self.total_cost}."
        ]
        for gene_node in self.gene_tree.find_clades(order="preorder"):
            species_node = self.gene_mapping[gene_node]
            line = (
                f"  {self._label(gene_node)} -> {self._label(species_node)} "
                f"at time {self.event_times[gene_node]}"
            )
            if not gene_node.is_terminal():
                event = self.events[gene_node]
                if event == TRANSFER:
                    target = self.transfer_targets[gene_node]
                    line += f": {event} to {self._label(target)}"
                else:
                    line += f": {event}"
            lines.append(line)
        for loss in self.losses:
            lines.append(
                "  loss on edge "
                f"{self._label(loss.gene_parent)}->{self._label(loss.gene_child)}: "
                f"lineage {self._label(loss.lost_species)} absent at "
                f"{self._label(loss.species_node)}"
            )
        return "\n".join(lines)


class _SpeciesIndex:
    """Array-based view of the dated, binary rooted species tree (PRIVATE)."""

    def __init__(self, species_tree, time_map):
        """Build index arrays and resolve node times (PRIVATE)."""
        self.tree = species_tree
        postorder = list(species_tree.find_clades(order="postorder"))
        preorder = list(species_tree.find_clades(order="preorder"))
        self.nodes = postorder
        self.node_id = {node: i for i, node in enumerate(postorder)}
        size = len(postorder)

        self.parent = [-1] * size
        self.left = [-1] * size
        self.right = [-1] * size
        self.depth = [0] * size
        for node_id, node in enumerate(postorder):
            if not node.is_terminal():
                left_id = self.node_id[node.clades[0]]
                right_id = self.node_id[node.clades[1]]
                self.left[node_id] = left_id
                self.right[node_id] = right_id
                self.parent[left_id] = node_id
                self.parent[right_id] = node_id
        for node in preorder:
            node_id = self.node_id[node]
            for child in node.clades:
                self.depth[self.node_id[child]] = self.depth[node_id] + 1

        self.name_to_node = {}
        for node in postorder:
            if node.name is not None:
                name = str(node.name)
                if name in self.name_to_node and self.name_to_node[name] is not node:
                    raise ValueError(
                        f"duplicate species-tree node name {name!r}; node names "
                        "must be unique to resolve the leaf map or time map"
                    )
                self.name_to_node[name] = node

        self.time = self._resolve_times(time_map, preorder)

        # Group node ids by time stamp for contemporaneous transfers.
        self.levels = {}
        for node_id, node_time in enumerate(self.time):
            self.levels.setdefault(node_time, []).append(node_id)

    def resolve_node(self, value):
        """Resolve a species-tree node given as a Clade instance or name."""
        if isinstance(value, BaseTree.Clade):
            if value not in self.node_id:
                raise ValueError("species node is not part of the species tree")
            return value
        name = str(value)
        try:
            return self.name_to_node[name]
        except KeyError:
            raise ValueError(f"unknown species-tree node name {name!r}") from None

    def _resolve_times(self, time_map, preorder):
        """Resolve and validate the time of every species-tree node."""
        size = len(self.nodes)
        if time_map is not None:
            times = [None] * size
            for key, value in time_map.items():
                node = self.resolve_node(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"time for node {self._node_label(node)} must be numeric"
                    )
                times[self.node_id[node]] = value
            if any(value is None for value in times):
                missing = [
                    self._node_label(node)
                    for node in preorder
                    if times[self.node_id[node]] is None
                ]
                raise ValueError(
                    "time_map must cover every species-tree node; missing: "
                    + ", ".join(missing)
                )
        elif all(getattr(node, "time", None) is not None for node in preorder):
            times = []
            for node in self.nodes:
                value = node.time
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"time attribute of node {self._node_label(node)} "
                        "must be numeric"
                    )
                times.append(value)
        else:
            times = self._times_from_branch_lengths(preorder)

        # Time must strictly decrease from parent to child.
        for node_id in range(size):
            parent_id = self.parent[node_id]
            if parent_id != -1 and not times[parent_id] > times[node_id]:
                raise ValueError(
                    "species-tree times must strictly decrease from parent to "
                    f"child (node {self._node_label(self.nodes[node_id])} has "
                    f"time {times[node_id]} not below parental time "
                    f"{times[parent_id]})"
                )
        return times

    def _times_from_branch_lengths(self, preorder):
        """Derive node times from cumulative branch lengths (ultrametric)."""
        cumulative = {}
        cumulative[preorder[0]] = 0
        for node in preorder:
            for child in node.clades:
                branch_length = child.branch_length
                if not isinstance(branch_length, (int, float)) or isinstance(
                    branch_length, bool
                ):
                    raise ValueError(
                        "no time_map or node 'time' attributes were supplied and "
                        "branch lengths are missing or non-numeric; cannot "
                        "derive species-tree node times"
                    )
                if branch_length <= 0:
                    raise ValueError(
                        "cannot derive strictly decreasing node times from "
                        "non-positive branch lengths; please supply an explicit "
                        "time_map"
                    )
                cumulative[child] = cumulative[node] + branch_length
        leaf_depths = {cumulative[node] for node in self.nodes if node.is_terminal()}
        if len(leaf_depths) != 1:
            raise ValueError(
                "species tree is not ultrametric from branch lengths; please "
                "supply an explicit time_map"
            )
        height = leaf_depths.pop()
        return [height - cumulative[node] for node in self.nodes]

    @staticmethod
    def _node_label(node):
        """Return a readable node label for error messages (PRIVATE)."""
        if node.name is not None:
            return repr(str(node.name))
        return node.__class__.__name__


def _check_binary_rooted(tree, what):
    """Validate that a tree is rooted and strictly binary (PRIVATE)."""
    if not getattr(tree, "rooted", True):
        raise ValueError(f"{what} tree must be rooted")
    for node in tree.find_clades():
        if node.is_terminal():
            continue
        if len(node.clades) != 2:
            label = repr(node.name) if node.name is not None else "an internal node"
            raise ValueError(
                f"{what} tree must be strictly bifurcating; {label} has "
                f"{len(node.clades)} children"
            )


def _resolve_leaf_map(gene_tree, index, leaf_map):
    """Resolve the gene-leaf to species-leaf map to node objects (PRIVATE)."""
    gene_leaves = gene_tree.get_terminals()
    name_to_gene = {}
    for leaf in gene_leaves:
        if leaf.name is not None:
            name_to_gene[str(leaf.name)] = leaf

    resolved = {}
    seen = set()
    gene_leaf_set = set(gene_leaves)
    return_items = (
        list(leaf_map.items()) if isinstance(leaf_map, dict) else list(leaf_map)
    )
    for key, value in return_items:
        if isinstance(key, BaseTree.Clade):
            gene_leaf = key
            if gene_leaf not in gene_leaf_set:
                raise ValueError("leaf map key is not a terminal of the gene tree")
        else:
            name = str(key)
            if name not in name_to_gene:
                raise ValueError(f"unknown gene-tree leaf name {name!r}")
            gene_leaf = name_to_gene[name]
        species_leaf = index.resolve_node(value)
        if not species_leaf.is_terminal():
            raise ValueError(
                "leaf map values must be terminal nodes of the species tree; "
                f"{index._node_label(species_leaf)} is internal"
            )
        if gene_leaf in seen:
            raise ValueError(
                "gene leaf appears more than once in the leaf map: "
                f"{index._node_label(gene_leaf)}"
            )
        seen.add(gene_leaf)
        resolved[gene_leaf] = species_leaf

    missing = [leaf for leaf in gene_leaves if leaf not in seen]
    if missing:
        labels = [index._node_label(leaf) for leaf in missing]
        raise ValueError(
            "every gene-tree leaf must be mapped to a species-tree leaf; "
            "missing: " + ", ".join(labels)
        )
    return resolved


def _placement_minima(cost_row, index, loss_cost):
    """Subtree placement minima with loss-adjusted descent (PRIVATE).

    Returns four arrays of length ``n`` for one gene node:

    - ``place_val[s]``: minimum of ``M[g, x] + L * dist(s, x)`` over
      ``x`` in the subtree of ``s``.
    - ``place_arg[s]``: the species id ``x`` attaining ``place_val[s]``.
    - ``retain_val[s]``: minimum cost for a child lineage born at ``s`` and
      staying in the subtree of ``s``, i.e. allowing ``x == s`` with no edge
      loss and descendants with the speciation nodes crossed in between.
    - ``retain_arg[s]``: the species id attaining ``retain_val[s]``.
    """
    size = len(cost_row)
    place_val = [_INF] * size
    place_arg = [-1] * size
    retain_val = [_INF] * size
    retain_arg = [-1] * size
    # The species id order is postorder, so children are handled first.
    for species_id in range(size):
        direct = cost_row[species_id]
        best_place_val = direct
        best_place_arg = species_id
        best_child_place_val = _INF
        best_child_place_arg = -1
        left_id = index.left[species_id]
        right_id = index.right[species_id]
        for child_id in (left_id, right_id):
            if child_id == -1 or place_val[child_id] >= _INF:
                continue
            candidate = loss_cost + place_val[child_id]
            if candidate < best_place_val:
                best_place_val = candidate
                best_place_arg = place_arg[child_id]
            if place_val[child_id] < best_child_place_val:
                best_child_place_val = place_val[child_id]
                best_child_place_arg = place_arg[child_id]
        place_val[species_id] = best_place_val
        place_arg[species_id] = best_place_arg
        if direct <= best_child_place_val:
            retain_val[species_id] = direct
            retain_arg[species_id] = species_id
        else:
            retain_val[species_id] = best_child_place_val
            retain_arg[species_id] = best_child_place_arg
    return place_val, place_arg, retain_val, retain_arg


def _level_minima(cost_row, index):
    """Best and second-best cost per time level for transfer targets (PRIVATE).

    Returns arrays ``level_val[s]`` and ``level_arg[s]`` giving the minimum of
    ``M[g, b]`` over species nodes ``b`` with the same time stamp as ``s`` but
    ``b != s`` (at a given time level only ``s`` itself can belong to the
    subtree of ``s``, so this is exactly the set of valid transfer targets).
    """
    size = len(cost_row)
    level_val = [_INF] * size
    level_arg = [-1] * size
    for ids in index.levels.values():
        ordered_ids = sorted(ids)
        best_val = _INF
        best_id = -1
        second_val = _INF
        second_id = -1
        for species_id in ordered_ids:
            value = cost_row[species_id]
            if value < best_val:
                second_val = best_val
                second_id = best_id
                best_val = value
                best_id = species_id
            elif value < second_val:
                second_val = value
                second_id = species_id
        for species_id in ordered_ids:
            if species_id == best_id:
                level_val[species_id] = second_val
                level_arg[species_id] = second_id
            else:
                level_val[species_id] = best_val
                level_arg[species_id] = best_id
    return level_val, level_arg


def reconcile_dtl(
    gene_tree,
    species_tree,
    leaf_map,
    duplication_cost=2,
    transfer_cost=3,
    loss_cost=1,
    time_map=None,
):
    """Reconcile a binary rooted gene tree with a dated binary species tree.

    Compute a most-parsimonious Duplication-Transfer-Loss reconciliation using
    the dated-tree dynamic program of Bansal, Alm and Kellis (2012).  The
    species tree must be fully dated (time strictly decreases along every
    root-to-leaf path), which restricts transfers to contemporaneous species
    nodes and therefore guarantees that the returned reconciliation is
    globally time-consistent.  Both input trees are treated as read-only.

    The dynamic program runs in O(mn) time and O(mn) memory for a gene tree
    with m nodes and a species tree with n nodes.  Ties are resolved
    deterministically (speciation before duplication before transfer, then the
    lower species-tree index), so repeated calls return the same result.

    :param gene_tree: binary rooted ``Tree`` whose leaves are gene copies.
        Several gene leaves may map to the same species leaf.  Newick trees
        are read as unrooted by default; pass ``rooted=True`` to
        ``Bio.Phylo.read`` (or set ``tree.rooted = True``) for rooted input.
    :param species_tree: binary rooted ``Tree`` whose nodes all carry times.
    :param leaf_map: mapping of each gene-tree leaf to a species-tree leaf.
        Keys and values may be ``Clade`` objects or node names (strings);
        several keys may share one value.
    :param duplication_cost: positive cost of a duplication event (default 2).
    :param transfer_cost: positive cost of a horizontal transfer (default 3).
    :param loss_cost: positive cost of a gene loss (default 1).
    :param time_map: optional dict mapping every species-tree node (``Clade``
        or name) to a numeric time.  When omitted, node ``time`` attributes
        are used if present; otherwise times are derived from branch lengths,
        which requires an ultrametric tree with positive branch lengths.
    :returns: a :class:`DTLReconciliation` with the optimal node mapping,
        event types and times, transfer targets, the explicit loss
        explanations and the minimum total cost.
    :raises ValueError: if the inputs are not rooted binary trees, costs are
        not positive, the leaf map is incomplete or unknown, or node times are
        missing or not strictly decreasing.
    """
    _check_binary_rooted(gene_tree, "gene")
    _check_binary_rooted(species_tree, "species")
    for name, cost in (
        ("duplication_cost", duplication_cost),
        ("transfer_cost", transfer_cost),
        ("loss_cost", loss_cost),
    ):
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            raise ValueError(f"{name} must be a positive number, got {cost!r}")

    index = _SpeciesIndex(species_tree, time_map)
    resolved_leaf_map = _resolve_leaf_map(gene_tree, index, leaf_map)
    size = len(index.nodes)

    # cost_table[g][s] and choice_table[g][s] for every gene node g.
    cost_table = {}
    choice_table = {}
    gene_postorder = list(gene_tree.find_clades(order="postorder"))

    for gene_node in gene_postorder:
        if gene_node.is_terminal():
            row = [_INF] * size
            species_leaf = resolved_leaf_map[gene_node]
            row[index.node_id[species_leaf]] = 0
            cost_table[gene_node] = row
            continue

        first_child, second_child = gene_node.clades
        first_cost = cost_table[first_child]
        second_cost = cost_table[second_child]
        (
            place1_val,
            place1_arg,
            retain1_val,
            retain1_arg,
        ) = _placement_minima(first_cost, index, loss_cost)
        (
            place2_val,
            place2_arg,
            retain2_val,
            retain2_arg,
        ) = _placement_minima(second_cost, index, loss_cost)
        level1_val, level1_arg = _level_minima(first_cost, index)
        level2_val, level2_arg = _level_minima(second_cost, index)

        row = [_INF] * size
        choices = [None] * size
        for species_id in range(size):
            best_value = _INF
            best_choice = None

            # Speciation: the two child lineages follow the two daughters.
            left_id = index.left[species_id]
            right_id = index.right[species_id]
            if left_id != -1:
                if place1_val[left_id] < _INF and place2_val[right_id] < _INF:
                    candidate = place1_val[left_id] + place2_val[right_id]
                    if candidate < best_value:
                        best_value = candidate
                        best_choice = (
                            _CHOICE_S,
                            place1_arg[left_id],
                            place2_arg[right_id],
                            -1,
                        )
                if place2_val[left_id] < _INF and place1_val[right_id] < _INF:
                    candidate = place2_val[left_id] + place1_val[right_id]
                    if candidate < best_value:
                        best_value = candidate
                        best_choice = (
                            _CHOICE_S,
                            place1_arg[right_id],
                            place2_arg[left_id],
                            -1,
                        )

            # Duplication: both child lineages stay below the same species.
            if retain1_val[species_id] < _INF and retain2_val[species_id] < _INF:
                candidate = (
                    duplication_cost + retain1_val[species_id] + retain2_val[species_id]
                )
                if candidate < best_value:
                    best_value = candidate
                    best_choice = (
                        _CHOICE_D,
                        retain1_arg[species_id],
                        retain2_arg[species_id],
                        -1,
                    )

            # Transfer: one child stays, the other jumps to a same-time node.
            if retain1_val[species_id] < _INF and level2_val[species_id] < _INF:
                candidate = (
                    transfer_cost + retain1_val[species_id] + level2_val[species_id]
                )
                if candidate < best_value:
                    best_value = candidate
                    best_choice = (
                        _CHOICE_T,
                        retain1_arg[species_id],
                        level2_arg[species_id],
                        2,
                    )
            if retain2_val[species_id] < _INF and level1_val[species_id] < _INF:
                candidate = (
                    transfer_cost + retain2_val[species_id] + level1_val[species_id]
                )
                if candidate < best_value:
                    best_value = candidate
                    best_choice = (
                        _CHOICE_T,
                        level1_arg[species_id],
                        retain2_arg[species_id],
                        1,
                    )

            row[species_id] = best_value
            choices[species_id] = best_choice
        cost_table[gene_node] = row
        choice_table[gene_node] = choices

    root_row = cost_table[gene_tree.root]
    root_species_id = min(range(size), key=lambda species_id: root_row[species_id])
    if root_row[root_species_id] >= _INF:
        raise RuntimeError("no feasible reconciliation exists for the given inputs")
    total_cost = root_row[root_species_id]

    # Top-down backtracking to recover mappings, events, times and losses.
    gene_mapping = {}
    events = {}
    event_times = {}
    transfer_targets = {}
    losses = []
    event_labels = {
        _CHOICE_S: SPECIATION,
        _CHOICE_D: DUPLICATION,
        _CHOICE_T: TRANSFER,
    }
    stack = [(gene_tree.root, root_species_id)]
    while stack:
        gene_node, species_id = stack.pop()
        gene_mapping[gene_node] = index.nodes[species_id]
        event_times[gene_node] = index.time[species_id]
        if gene_node.is_terminal():
            continue

        choice_code, first_target, second_target, transferred = choice_table[gene_node][
            species_id
        ]
        events[gene_node] = event_labels[choice_code]
        first_child, second_child = gene_node.clades
        if choice_code == _CHOICE_T:
            target_id = second_target if transferred == 2 else first_target
            transfer_targets[gene_node] = index.nodes[target_id]

        for child, target_id, is_transferred in (
            (first_child, first_target, transferred == 1),
            (second_child, second_target, transferred == 2),
        ):
            if not is_transferred:
                # Each speciation node strictly between the mapped species of
                # parent and child loses the sister species branch.
                current_id = target_id
                while current_id != species_id:
                    parent_id = index.parent[current_id]
                    if parent_id != species_id:
                        if index.left[parent_id] == current_id:
                            lost_id = index.right[parent_id]
                        else:
                            lost_id = index.left[parent_id]
                        losses.append(
                            Loss(
                                gene_parent=gene_node,
                                gene_child=child,
                                species_node=index.nodes[parent_id],
                                lost_species=index.nodes[lost_id],
                            )
                        )
                    current_id = parent_id
            stack.append((child, target_id))

    return DTLReconciliation(
        gene_tree=gene_tree,
        species_tree=species_tree,
        gene_mapping=gene_mapping,
        events=events,
        event_times=event_times,
        transfer_targets=transfer_targets,
        losses=losses,
        total_cost=total_cost,
        duplication_cost=duplication_cost,
        transfer_cost=transfer_cost,
        loss_cost=loss_cost,
    )
