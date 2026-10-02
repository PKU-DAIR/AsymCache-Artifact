import unittest

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams


class StubPenalizerOrchestrator:
    def filter(self, keep_indices):
        pass

    def merge(self, other):
        pass


def make_sampling_info(std_output_tokens=None, next_indices=None, eos_token_ids=None):
    batch_size = len(std_output_tokens) if std_output_tokens is not None else 1
    return SamplingBatchInfo(
        temperatures=torch.ones((batch_size, 1)),
        top_ps=torch.ones(batch_size),
        top_ks=torch.ones(batch_size, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=10,
        penalizer_orchestrator=StubPenalizerOrchestrator(),
        device="cpu",
        std_output_tokens=std_output_tokens,
        std_output_tokens_next_idx=next_indices,
        std_output_eos_token_ids=eos_token_ids,
    )


class TestStdOutputTokens(unittest.TestCase):
    def test_sampling_params_validation_and_normalization(self):
        params = SamplingParams(std_output_tokens=[0, 9])
        params.verify(vocab_size=10)
        params.normalize(type("Tokenizer", (), {"eos_token_id": 8})())
        self.assertEqual(params.eos_token_id, 8)

        for invalid_tokens in ([True], [-1], [10], [1.5], "1,2"):
            with self.subTest(invalid_tokens=invalid_tokens):
                with self.assertRaises(ValueError):
                    SamplingParams(std_output_tokens=invalid_tokens).verify(vocab_size=10)

    def test_forced_tokens_advance_and_end_with_eos(self):
        sampling_info = make_sampling_info(
            std_output_tokens=[[4, 5], None, []],
            next_indices=[[0], [0], [0]],
            eos_token_ids=[9, None, 8],
        )

        sampled = torch.tensor([1, 2, 3])
        self.assertEqual(
            sampling_info.apply_std_output_tokens(sampled).tolist(), [4, 2, 8]
        )
        self.assertEqual(
            sampling_info.apply_std_output_tokens(sampled).tolist(), [5, 2, 8]
        )
        self.assertEqual(
            sampling_info.apply_std_output_tokens(sampled).tolist(), [9, 2, 8]
        )

    def test_filter_preserves_each_request_index_cell(self):
        first_index = [1]
        third_index = [2]
        sampling_info = make_sampling_info(
            std_output_tokens=[[1, 2], [3, 4], [5, 6, 7]],
            next_indices=[first_index, [0], third_index],
            eos_token_ids=[8, 8, 8],
        )

        sampling_info.filter_batch([2, 0], torch.tensor([2, 0]))

        self.assertEqual(sampling_info.std_output_tokens, [[5, 6, 7], [1, 2]])
        self.assertIs(sampling_info.std_output_tokens_next_idx[0], third_index)
        self.assertIs(sampling_info.std_output_tokens_next_idx[1], first_index)

    def test_merge_supports_mixed_forced_and_normal_requests(self):
        normal = make_sampling_info()
        forced = make_sampling_info(
            std_output_tokens=[[7]],
            next_indices=[[0]],
            eos_token_ids=[9],
        )

        normal.merge_batch(forced)

        self.assertEqual(normal.apply_std_output_tokens(torch.tensor([1, 2])).tolist(), [1, 7])


if __name__ == "__main__":
    unittest.main()