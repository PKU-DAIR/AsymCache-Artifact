def update_cached_tokens(
    cached_tokens: int, already_computed: int, prefix_len: int, seq_len: int
) -> tuple[int, int]:
    return cached_tokens + prefix_len - already_computed, seq_len