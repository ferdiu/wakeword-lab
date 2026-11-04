from __future__ import annotations

from collections import defaultdict
import random


def _label_counts(rows: list[dict]) -> tuple[int, int]:
    positives = sum(row.get("label") == 1 for row in rows)
    negatives = sum(row.get("label") == 0 for row in rows)
    return positives, negatives


def group_split(records, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Split by source group, respecting explicit pins and approximately stratifying labels.

    A source group is indivisible: recordings from the same person/device/room cannot
    leak across train, validation and test. When labels are present, the allocator first
    ensures every partition gets a positive and negative group where the available
    independent groups make that possible, then greedily balances class/group counts.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-8 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to one")
    records = list(records)
    groups: dict[str, list] = defaultdict(list)
    for row in records:
        group = row.get("source_group")
        if not group:
            raise ValueError("Every row requires a non-empty source_group")
        groups[str(group)].append(row)

    names = ("train", "validation", "test")
    total_class_counts = _label_counts(records)
    has_labels = bool(sum(total_class_counts))
    pinned: dict[str, list] = {name: [] for name in names}
    pinned_group_counts = {name: 0 for name in names}
    free: list[tuple[str, list]] = []
    for key, rows in groups.items():
        choices = {r.get("split") for r in rows if r.get("split")}
        if len(choices) > 1:
            raise ValueError(f"source_group {key!r} has conflicting split pins: {choices}")
        choice = next(iter(choices), None)
        if choice is not None:
            if choice not in names:
                raise ValueError(f"Invalid pinned split {choice!r} for group {key!r}")
            pinned[choice].extend(rows)
            pinned_group_counts[choice] += int(bool(sum(_label_counts(rows))) if has_labels else 1)
        else:
            free.append((key, rows))

    rng = random.Random(seed)
    rng.shuffle(free)
    allocated = {name: list(pinned[name]) for name in names}
    split_group_counts = dict(pinned_group_counts)
    split_class_counts = {name: list(_label_counts(allocated[name])) for name in names}
    total_label_groups = (sum(bool(sum(_label_counts(rows))) for rows in groups.values())
                          if has_labels else len(groups))
    target_groups = [total_label_groups * fraction for fraction in fractions]
    target_classes = {
        name: [total_class_counts[0] * fraction, total_class_counts[1] * fraction]
        for name, fraction in zip(names, fractions)
    }

    # Assign groups that satisfy currently missing classes before general balancing.
    # Choose the most constrained (fewest remaining candidate groups) task first.
    remaining = list(free)
    while has_labels:
        tasks = []
        for name in names:
            for class_index in (0, 1):
                if split_class_counts[name][class_index] > 0:
                    continue
                candidates = [item for item in remaining if _label_counts(item[1])[class_index] > 0]
                if candidates:
                    tasks.append((len(candidates), name, class_index, candidates))
        if not tasks:
            break
        _, target_name, class_index, candidates = min(tasks, key=lambda task: task[0])

        def assignment_cost(item):
            counts = _label_counts(item[1])
            current = split_class_counts[target_name]
            class_delta = sum(
                (((current[i] + counts[i] - target_classes[target_name][i]) ** 2)
                 - ((current[i] - target_classes[target_name][i]) ** 2))
                / max(target_classes[target_name][i], 1.0)
                for i in (0, 1)
            )
            adds_group = int(bool(sum(counts)))
            group_delta = 0.25 * (
                ((split_group_counts[target_name] + adds_group - target_groups[names.index(target_name)]) ** 2
                 - (split_group_counts[target_name] - target_groups[names.index(target_name)]) ** 2)
                / max(target_groups[names.index(target_name)], 1.0)
            )
            # Prefer a group that also fills the other missing class in this partition.
            other = 1 - class_index
            bonus = -0.5 if counts[other] > 0 and split_class_counts[target_name][other] == 0 else 0.0
            return class_delta + group_delta + bonus

        selected = min(candidates, key=assignment_cost)
        key, rows = selected
        allocated[target_name].extend(rows)
        counts = _label_counts(rows)
        split_class_counts[target_name][0] += counts[0]
        split_class_counts[target_name][1] += counts[1]
        split_group_counts[target_name] += int(bool(sum(counts)))
        remaining.remove(selected)

    # Greedily place the remaining groups at the partition with the smallest normalized
    # change in class and labelled-group count error.
    remaining.sort(key=lambda item: len(item[1]), reverse=True)
    for key, rows in remaining:
        counts = _label_counts(rows)
        labelled_group = int(bool(sum(counts))) if has_labels else 1
        costs = []
        for index, name in enumerate(names):
            class_delta = sum(
                (((split_class_counts[name][i] + counts[i] - target_classes[name][i]) ** 2)
                 - ((split_class_counts[name][i] - target_classes[name][i]) ** 2))
                / max(target_classes[name][i], 1.0)
                for i in (0, 1)
            )
            group_delta = 0.25 * (
                ((split_group_counts[name] + labelled_group - target_groups[index]) ** 2
                 - (split_group_counts[name] - target_groups[index]) ** 2)
                / max(target_groups[index], 1.0)
            )
            costs.append(class_delta + group_delta)
        selected_name = names[min(range(len(names)), key=lambda i: costs[i])]
        allocated[selected_name].extend(rows)
        split_class_counts[selected_name][0] += counts[0]
        split_class_counts[selected_name][1] += counts[1]
        split_group_counts[selected_name] += labelled_group

    if any(not allocated[name] for name in names):
        raise ValueError("Need independent source groups for train, validation and test; add recordings or metadata split pins")
    return allocated
