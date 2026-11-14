"""Tests for AudioLLM vocabulary mapping, conditioning and generation."""

from __future__ import annotations

import pytest
import torch

from apollo_voice_engine.models.audio_llm import AudioLLM


@pytest.fixture
def unloaded() -> AudioLLM:
    model = AudioLLM(device="cpu")
    model.configure_vocabulary(68098)
    return model


class TestIdMapping:
    def test_requires_configuration(self):
        with pytest.raises(RuntimeError, match="load_model"):
            AudioLLM(device="cpu").audio_tokens_to_ids(torch.tensor([1]))

    def test_rejects_conflicting_quantization(self):
        with pytest.raises(ValueError):
            AudioLLM(load_in_8bit=True, load_in_4bit=True)

    def test_raw_and_offset_tokens_map_identically(self, unloaded):
        raw = torch.tensor([0, 1, 4095])
        assert unloaded.audio_tokens_to_ids(raw).tolist() == [68098, 68099, 72193]
        assert torch.equal(
            unloaded.audio_tokens_to_ids(raw + 50000), unloaded.audio_tokens_to_ids(raw)
        )

    def test_audio_ids_do_not_collide_with_text_vocab(self, unloaded):
        # Sarvam-1's text vocabulary is larger than the legacy 50000 offset.
        ids = unloaded.audio_tokens_to_ids(torch.tensor([0]))
        assert ids.item() >= unloaded.audio_offset > 50000

    def test_out_of_range_tokens_raise(self, unloaded):
        with pytest.raises(ValueError):
            unloaded.audio_tokens_to_ids(torch.tensor([4096]))
        with pytest.raises(ValueError):
            unloaded.audio_tokens_to_ids(torch.tensor([-1]))

    def test_ids_to_audio_tokens_inverse(self, unloaded):
        snac_tokens = torch.tensor([50000, 52000, 54095])
        ids = unloaded.audio_tokens_to_ids(snac_tokens)
        assert torch.equal(unloaded.ids_to_audio_tokens(ids), snac_tokens)

    def test_is_audio_token_bounds(self, unloaded):
        assert not unloaded.is_audio_token(68097)
        assert unloaded.is_audio_token(68098)
        assert unloaded.is_audio_token(72193)
        assert not unloaded.is_audio_token(72194)


class TestLoadedModel:
    def test_vocabulary_extended(self, audio_llm):
        assert audio_llm.audio_offset == len(audio_llm.tokenizer)
        assert audio_llm.extended_vocab_size == audio_llm.audio_offset + 4096
        assert (
            audio_llm.model.get_input_embeddings().weight.shape[0] == audio_llm.extended_vocab_size
        )
        assert audio_llm.audio_start_id < audio_llm.audio_offset
        assert audio_llm.audio_end_id < audio_llm.audio_offset

    def test_save_and_reload_does_not_re_extend(self, audio_llm, tmp_path):
        audio_llm.save_pretrained(str(tmp_path))
        reloaded = AudioLLM.from_pretrained(str(tmp_path), device="cpu")
        assert reloaded.audio_offset == audio_llm.audio_offset
        assert reloaded.extended_vocab_size == audio_llm.extended_vocab_size
        assert reloaded.audio_start_id == audio_llm.audio_start_id

    def test_build_audio_input_ids_layout(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([[50000, 50005]]), "hello world")
        seq = ids[0].tolist()
        hello, world = audio_llm.tokenizer.convert_tokens_to_ids(["hello", "world"])
        offset = audio_llm.audio_offset
        assert seq == [
            hello,
            world,
            audio_llm.audio_start_id,
            offset,
            offset + 5,
            audio_llm.audio_end_id,
        ]

    def test_tokens_to_text_skips_audio(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([1, 2, 3]), "hello world")
        assert audio_llm.tokens_to_text(ids[0]) == "hello world"

    def test_prepare_audio_input_without_speaker(self, audio_llm):
        embeds, mask = audio_llm.prepare_audio_input(torch.tensor([1, 2, 3]), "hello")
        assert embeds.shape == (1, 1 + 1 + 3 + 1, audio_llm.model.config.hidden_size)
        assert mask.shape == embeds.shape[:2]
        assert mask.sum() == mask.numel()

    def test_prepare_audio_input_prepends_speaker_prefix(self, audio_llm):
        speaker = torch.randn(1, AudioLLM.SPEAKER_DIM)
        plain, _ = audio_llm.prepare_audio_input(torch.tensor([1, 2, 3]), "hello")
        conditioned, mask = audio_llm.prepare_audio_input(
            torch.tensor([1, 2, 3]), "hello", speaker_embedding=speaker
        )
        assert conditioned.shape[1] == plain.shape[1] + 1
        assert torch.allclose(conditioned[:, 1:], plain)
        expected_prefix = audio_llm.speaker_proj(speaker)
        assert torch.allclose(conditioned[:, 0], expected_prefix)
        assert mask.shape == conditioned.shape[:2]

    def test_generate_from_embeddings(self, audio_llm):
        embeds, mask = audio_llm.prepare_audio_input(
            torch.tensor([1, 2, 3]), speaker_embedding=torch.randn(1, 512)
        )
        out = audio_llm.generate_response(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=4,
            min_new_tokens=4,
            do_sample=False,
        )
        assert out.shape == (1, 4)  # embeddings input -> only new tokens returned
        assert out.max() < audio_llm.extended_vocab_size

    def test_generate_from_ids_includes_prompt(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([1, 2]))
        out = audio_llm.generate_response(
            input_ids=ids, max_new_tokens=3, min_new_tokens=3, do_sample=False
        )
        assert out.shape == (1, ids.shape[1] + 3)
        assert torch.equal(out[:, : ids.shape[1]], ids)

    def test_generate_requires_input(self, audio_llm):
        with pytest.raises(ValueError):
            audio_llm.generate_response()

    def test_generate_override_short_circuits(self, audio_llm):
        override = torch.tensor([[50000, 50001]])
        assert torch.equal(audio_llm.generate_response(override_output_ids=override), override)

    def test_forward_computes_loss(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([1, 2, 3]), "hello")
        out = audio_llm(input_ids=ids, labels=ids)
        assert torch.isfinite(out.loss)


class TestExtractAudioTokens:
    def test_segment_between_markers(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([7, 8, 9]), "hello")
        assert audio_llm.extract_audio_tokens(ids).tolist() == [[50007, 50008, 50009]]

    def test_uses_last_start_marker(self, audio_llm):
        prompt = audio_llm.build_audio_input_ids(torch.tensor([1, 1, 1]))
        reply = audio_llm.build_audio_input_ids(torch.tensor([5, 6]))
        tokens = audio_llm.extract_audio_tokens(torch.cat([prompt, reply], dim=1))
        assert tokens.tolist() == [[50005, 50006]]

    def test_reads_to_end_without_end_marker(self, audio_llm):
        ids = audio_llm.build_audio_input_ids(torch.tensor([3, 4]))[:, :-1]
        assert audio_llm.extract_audio_tokens(ids).tolist() == [[50003, 50004]]

    def test_bare_audio_ids_without_markers(self, audio_llm):
        ids = audio_llm.audio_tokens_to_ids(torch.tensor([[2, 3]]))
        assert audio_llm.extract_audio_tokens(ids).tolist() == [[50002, 50003]]

    def test_filters_text_tokens_inside_segment(self, audio_llm):
        hello = audio_llm.tokenizer.convert_tokens_to_ids("hello")
        offset = audio_llm.audio_offset
        ids = torch.tensor([[audio_llm.audio_start_id, offset + 1, hello, offset + 2]])
        assert audio_llm.extract_audio_tokens(ids).tolist() == [[50001, 50002]]

    def test_returns_none_without_audio(self, audio_llm):
        hello = audio_llm.tokenizer.convert_tokens_to_ids("hello")
        assert audio_llm.extract_audio_tokens(torch.tensor([[hello, hello]])) is None
