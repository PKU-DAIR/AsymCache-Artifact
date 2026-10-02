import unittest

from sglang.srt.managers.cached_tokens import update_cached_tokens


class TestCachedTokens(unittest.TestCase):
    def test_chunked_prefill_does_not_count_previous_chunks_twice(self):
        cached_tokens = 0
        already_computed = 0

        cached_tokens, already_computed = update_cached_tokens(
            cached_tokens, already_computed, prefix_len=80, seq_len=100
        )
        self.assertEqual(cached_tokens, 80)

        cached_tokens, already_computed = update_cached_tokens(
            cached_tokens, already_computed, prefix_len=100, seq_len=120
        )
        self.assertEqual(cached_tokens, 80)

    def test_new_cache_hits_after_chunking_are_counted_incrementally(self):
        cached_tokens, already_computed = update_cached_tokens(
            cached_tokens=0,
            already_computed=0,
            prefix_len=80,
            seq_len=100,
        )
        cached_tokens, _ = update_cached_tokens(
            cached_tokens,
            already_computed,
            prefix_len=110,
            seq_len=130,
        )

        self.assertEqual(cached_tokens, 90)


if __name__ == "__main__":
    unittest.main()