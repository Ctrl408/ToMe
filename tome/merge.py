# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


import math
import random
from typing import Callable, Tuple

import torch
import torch.nn as nn


class LSMerger(nn.Module):
    def __init__(self, r: int = 8, num_buckets: int = 50, e_param: int = 1, similarity_metric: str = 'cosine'):
        super(LSHMerger, self).__init__()
        self.r = r
        self.num_buckets = num_buckets
        self.e_param = e_param
        self.similarity_metric = similarity_metric

    def _hash(self, x, projections):
        projections = projections.to(x.device)
        return (torch.einsum('btf,hf->bht', x, projections) > 0).float()

    def _l2_norm(self, x):
        return nn.functional.normalize(x, p=2, dim=-1)

    def _compute_similarity(self, bucket):
        if self.similarity_metric == 'dot_product':
            return torch.einsum('bif,bjf->bij', bucket, bucket)
        elif self.similarity_metric == 'cosine':
            norm_bucket = self._l2_norm(bucket)
            return torch.einsum('bif,bjf->bij', norm_bucket, norm_bucket)
        elif self.similarity_metric == 'softmax':
            similarity_matrix = torch.einsum('bif,bjf->bij', bucket, bucket)
            return torch.softmax(similarity_matrix, dim=-1)
        elif self.similarity_metric == 'angular':
            norm_bucket = self._l2_norm(bucket)
            cosine_similarity = torch.einsum('bif,bjf->bij', norm_bucket, norm_bucket)
            return 1 - torch.acos(cosine_similarity) / math.pi
        elif self.similarity_metric == 'pearson':
            mean = torch.mean(bucket, dim=-1, keepdim=True)
            bucket_centered = bucket - mean
            norm_bucket_centered = bucket_centered / bucket_centered.norm(dim=-1, keepdim=True)
            return torch.einsum('bif,bjf->bij', norm_bucket_centered, norm_bucket_centered)
        elif self.similarity_metric == 'euclidean':
            distances = torch.cdist(bucket, bucket, p=2)
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def bipartite_soft_matching(
        self, metric: torch.Tensor, r: int, class_token: bool = False, distill_token: bool = False
    ) -> Tuple[Callable, Callable]:
        protected = 0
        if class_token:
            protected += 1
        if distill_token:
            protected += 1

        t = metric.shape[1]
        r = min(r, (t - protected) // 2)

        if r <= 0:
            return do_nothing, do_nothing

        batch_size, num_tokens, num_features = metric.shape
        num_hashes = num_tokens
        projections = torch.randn(num_hashes, num_features, device=metric.device)

        metric = self._l2_norm(metric)
        hash_codes = self._hash(metric, projections)

        _, sorted_indices = hash_codes.sum(dim=-1).sort(dim=1)
        sorted_metric = metric.gather(dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, num_features))

        bucket_size = (num_tokens + self.num_buckets - 1) // self.num_buckets
        buckets = sorted_metric.split(bucket_size, dim=1)
        selected_indices = random.sample(range(len(buckets)), min(self.e_param, len(buckets)))
        merged_buckets = []

        for i, bucket in enumerate(buckets):
            if i in selected_indices and bucket.size(1) >= self.r:
                similarity_matrix = self._compute_similarity(bucket)
                similarity_scores, _ = similarity_matrix.triu(diagonal=1).max(dim=-1)
                top_r_indices = similarity_scores.topk(self.r, dim=1, largest=True).indices
                top_r_tokens = bucket.gather(dim=1, index=top_r_indices.unsqueeze(-1).expand(-1, -1, num_features))
                merged_token = top_r_tokens.mean(dim=1, keepdim=True)
                merged_buckets.append(merged_token)
                remaining_tokens = bucket[:, self.r:, :]
                merged_buckets.append(remaining_tokens)
            else:
                merged_buckets.append(bucket)

        merged_tokens = torch.cat(merged_buckets, dim=1)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            return merged_tokens

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            return sorted_metric

        return merge, unmerge




def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
