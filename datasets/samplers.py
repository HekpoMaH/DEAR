from clrs._src.samplers import (
    Sampler,
    _batch_hints,
    SAMPLERS,
    MatcherSampler,
    SortingSampler,
    SccSampler,
    SearchSampler,
)


class RNGMatcherSampler(Sampler):
    """String matching sampler; embeds needle in a random haystack."""

    def _sample_data(
        self,
        length: int,  # length of haystack + needle, i.e., total number of nodes
        length_needle=None,
        chars: int = 4,
    ):
        if length_needle is None:
            if length < 5:
                length_needle = 1
            else:
                length_needle = length // 5
        elif length_needle < 0:  # randomize needle length
            length_needle = self._rng.randint(1, high=1 - length_needle)
        length_haystack = length - length_needle
        needle = self._random_string(length=length_needle, chars=chars)
        haystack = self._random_string(length=length_haystack, chars=chars)
        embed_pos = self._rng.choice(length_haystack - length_needle)
        haystack[embed_pos : embed_pos + length_needle] = needle
        return [haystack, needle, self._rng]


class BetterMatcherSampler(Sampler):
    """String matching sampler; embeds needle in a random haystack."""

    def _sample_data(
        self,
        length: int,  # length of haystack + needle, i.e., total number of nodes
        length_needle=None,
        chars: int = 4,
    ):
        if length_needle is None:
            if length < 5:
                length_needle = 1
            else:
                length_needle = length // 5
        elif length_needle < 0:  # randomize needle length
            length_needle = self._rng.randint(1, high=1 - length_needle)
        length_haystack = length - length_needle
        needle = self._random_string(length=length_needle, chars=chars)
        haystack = self._random_string(length=length_haystack, chars=chars)
        if length_needle <= length_haystack:
            embed_pos = self._rng.randint(0, high=1 + length_haystack - length_needle)
            haystack[embed_pos : embed_pos + length_needle] = needle
        return [haystack, needle]


class BetterSearchSampler(Sampler):
    """Search sampler. Generates a random sequence and target (of U[0, 1])."""

    def _sample_data(
        self,
        length: int,
        low: float = 0.0,
        high: float = 1.0,
    ):
        arr = self._random_sequence(length=length, low=low, high=high)
        arr.sort()
        x = self._rng.uniform(low=low, high=high)
        if x > arr[-1]:
            x, arr[-1] = arr[-1], x
        return [x, arr]


LSAMPLERS = {
    "kmp_matcher": BetterMatcherSampler,
    "heapsort_local": SortingSampler,
    "insertion_sort_local": SortingSampler,
    "strongly_connected_components_local": SccSampler,
    "search": BetterSearchSampler,
    "binary_search": BetterSearchSampler,
}

SAMPLERS = SAMPLERS | LSAMPLERS
assert SAMPLERS["binary_search"] is BetterSearchSampler
