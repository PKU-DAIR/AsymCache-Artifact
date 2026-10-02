import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


class TestSchedulerOutputProcessor(unittest.TestCase):
    def run_finished_mixed_prefill(self, prompt_length):
        allocator = SimpleNamespace(free=Mock())
        scheduler = SimpleNamespace(
            is_generation=True,
            enable_overlap=True,
            is_mixed_chunk=True,
            page_size=64,
            tp_worker=SimpleNamespace(
                resolve_last_batch_result=Mock(return_value=(None, [42], None))
            ),
            token_to_kv_pool_allocator=allocator,
            set_next_batch_sampling_info_done=Mock(),
            stream_output=Mock(),
        )
        req = SimpleNamespace(
            is_retracted=False,
            origin_input_ids=[0] * prompt_length,
            output_ids=[42],
            finished=Mock(return_value=True),
        )
        batch = SimpleNamespace(
            reqs=[req],
            out_cache_loc=[99],
            return_logprob=False,
        )
        result = SimpleNamespace(
            logits_output=None,
            next_token_ids=None,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

        SchedulerOutputProcessorMixin.process_batch_result_prefill(
            scheduler, batch, result
        )
        return allocator.free

    def test_finished_delayed_token_within_page_is_not_freed(self):
        free = self.run_finished_mixed_prefill(prompt_length=1)

        free.assert_not_called()

    def test_finished_delayed_token_starting_new_page_is_freed(self):
        free = self.run_finished_mixed_prefill(prompt_length=64)

        free.assert_called_once_with([99])


if __name__ == "__main__":
    unittest.main()