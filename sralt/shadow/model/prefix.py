"""Shared item-path dictionary and deduplicated user-prefix sets."""

import torch


def prefix_dictionary(codes, levels, codebooks, item_embeddings=None):
    """Return globally numbered prefixes and their fixed initial representations.

    codes: [M,L]; levels are one-based. An explicit identity control supplies
    item_embeddings instead of reconstructing vectors from the RVQ codebooks.
    """
    maps, paths, level_ids, vectors = [], [], [], []
    offset = 0
    for level in levels:
        unique, inverse = torch.unique(codes[:, :level], dim=0, sorted=True, return_inverse=True)
        padded = codes.new_full((len(unique), codes.size(1)), -1)
        padded[:, :level] = unique
        if item_embeddings is None:
            reconstructed = sum(codebooks[l][unique[:, l]] for l in range(level))
        else:
            # Identity codes are consecutive local item rows at every level.
            reconstructed = item_embeddings[unique[:, 0]]
        maps.append(inverse + offset)
        paths.append(padded)
        level_ids.append(codes.new_full((len(unique),), level))
        vectors.append(reconstructed)
        offset += len(unique)
    return dict(item_prefix_ids=torch.stack(maps, dim=1), prefix_paths=torch.cat(paths),
                prefix_levels=torch.cat(level_ids), prefix_reconstruction=torch.cat(vectors))


def user_prefix_sets(edges, queried_item_ids, item_prefix_ids, num_users, num_prefixes, k):
    """Create [N,m,K] sets, sorted per user/layer with -1 padding.

    Graph edges use global item IDs offset by num_users. All inputs must refer
    to the same frozen dictionary; training edges are a subset of full queries.
    """
    local = torch.searchsorted(queried_item_ids, edges[1] - num_users)
    result = edges.new_full((num_users, item_prefix_ids.size(1), k), -1)
    for layer in range(item_prefix_ids.size(1)):
        keys = torch.unique(edges[0] * num_prefixes + item_prefix_ids[local, layer], sorted=True)
        users = torch.div(keys, num_prefixes, rounding_mode='floor')
        counts = torch.bincount(users, minlength=num_users)
        starts = counts.cumsum(0) - counts
        columns = torch.arange(len(keys), device=edges.device) - starts[users]
        result[users, layer, columns] = keys % num_prefixes
    return result


def prefix_frequency(prefix_ids, num_prefixes, frequency_weighting=True):
    active = prefix_ids[prefix_ids >= 0]
    frequency = torch.bincount(active, minlength=num_prefixes)
    information = ((prefix_ids.size(0) + 1.) / (frequency.double() + 1.)).log()
    if not frequency_weighting:
        information = torch.ones_like(information)
    return frequency, information.float()


def prefix_mass(prefix_ids, information):
    return (information[prefix_ids.clamp_min(0)] * (prefix_ids >= 0)).sum(-1)


def semantic_links(prefix_ids, information, node_offset):
    users = torch.arange(prefix_ids.size(0), device=prefix_ids.device)[:, None, None].expand_as(prefix_ids)
    valid = (prefix_ids >= 0) & (information[prefix_ids.clamp_min(0)] > 0)
    edges = torch.stack([users[valid], prefix_ids[valid] + node_offset])
    weights = information[prefix_ids[valid]]
    return torch.cat([edges, edges.flip(0)], dim=1), weights.repeat(2)
