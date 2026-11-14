"""Tests for SNACWrapper token layout and codec integration."""

from __future__ import annotations

import pytest
import torch

from apollo_voice_engine.models.snac_wrapper import SNACWrapper


def make_codes(batch: int, frames: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, 4096, (batch, frames * ratio), generator=g)
        for ratio in SNACWrapper.SCALE_RATIOS
    ]


class TestTokenLayout:
    def test_constants_match_snac_24khz(self):
        assert SNACWrapper.SCALE_RATIOS == (1, 2, 4)
        assert SNACWrapper.TOKENS_PER_FRAME == 7
        assert SNACWrapper.AUDIO_TOKEN_OFFSET == 50000
        assert SNACWrapper.AUDIO_VOCAB_SIZE == 4096

    def test_flatten_interleaves_frame_by_frame(self):
        wrapper = SNACWrapper(device="cpu")
        coarse = torch.tensor([[10, 20]])
        medium = torch.tensor([[11, 12, 21, 22]])
        fine = torch.tensor([[13, 14, 15, 16, 23, 24, 25, 26]])
        flat = wrapper._flatten_codes([coarse, medium, fine])
        assert flat.tolist() == [[10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26]]

    @pytest.mark.parametrize("batch,frames", [(1, 1), (1, 12), (3, 5)])
    def test_flatten_unflatten_roundtrip(self, batch, frames):
        wrapper = SNACWrapper(device="cpu")
        codes = make_codes(batch, frames)
        flat = wrapper._flatten_codes(codes)
        assert flat.shape == (batch, frames * 7)
        restored = wrapper._unflatten_codes(flat)
        for original, back in zip(codes, restored, strict=True):
            assert torch.equal(original, back)

    def test_unflatten_drops_partial_trailing_frame(self):
        wrapper = SNACWrapper(device="cpu")
        flat = wrapper._flatten_codes(make_codes(1, 2))
        codes = wrapper._unflatten_codes(torch.cat([flat, flat[:, :3]], dim=1))
        assert [c.shape[1] for c in codes] == [2, 4, 8]

    def test_unflatten_rejects_too_few_tokens(self):
        with pytest.raises(ValueError, match="at least 7"):
            SNACWrapper(device="cpu")._unflatten_codes(torch.zeros(1, 6, dtype=torch.long))

    def test_flatten_rejects_inconsistent_scales(self):
        codes = make_codes(1, 3)
        codes[2] = codes[2][:, :-1]
        with pytest.raises(ValueError, match="Inconsistent"):
            SNACWrapper(device="cpu")._flatten_codes(codes)

    def test_strip_offset_and_clamp(self):
        wrapper = SNACWrapper(device="cpu")
        assert wrapper._strip_offset(torch.tensor([50000, 54095])).tolist() == [0, 4095]
        assert wrapper._strip_offset(torch.tensor([0, 5000])).tolist() == [0, 4095]


class TestDurations:
    def test_frames_per_second(self):
        assert SNACWrapper().frames_per_second == pytest.approx(24000 / 2048)

    def test_token_count_matches_codec_padding(self):
        wrapper = SNACWrapper()
        assert wrapper.get_token_count(24000 / 24000) == 12 * 7  # 24000 samples -> 12 frames
        assert wrapper.get_token_count(2048 / 24000) == 7
        assert wrapper.get_token_count(0.0) == 0

    def test_audio_duration_roundtrip(self):
        wrapper = SNACWrapper()
        assert wrapper.get_audio_duration(7 * 24) == pytest.approx(24 * 2048 / 24000)
        assert wrapper.get_audio_duration(6) == 0.0


class TestShapes:
    @pytest.mark.parametrize("shape", [(24000,), (1, 24000), (1, 1, 24000)])
    def test_to_codec_shape(self, shape):
        assert SNACWrapper._to_codec_shape(torch.zeros(shape)).shape == (1, 1, 24000)

    def test_to_codec_shape_rejects_multichannel(self):
        with pytest.raises(ValueError, match="mono"):
            SNACWrapper._to_codec_shape(torch.zeros(1, 2, 100))


class TestWithCodec:
    """Run real SNAC encode/decode with a randomly initialised 24 kHz model."""

    @pytest.mark.parametrize("shape", [(24000,), (1, 24000), (1, 1, 24000)])
    def test_encode_accepts_common_shapes(self, snac, shape):
        tokens = snac.encode(torch.randn(shape) * 0.1)
        assert tokens.shape == (1, snac.get_token_count(1.0))
        assert tokens.min() >= SNACWrapper.AUDIO_TOKEN_OFFSET
        assert tokens.max() < SNACWrapper.AUDIO_TOKEN_OFFSET + SNACWrapper.AUDIO_VOCAB_SIZE

    def test_encode_matches_codec_codes(self, snac):
        audio = torch.randn(1, 1, 4096) * 0.1
        codes = snac.model.encode(audio)
        flat = snac.encode(audio) - SNACWrapper.AUDIO_TOKEN_OFFSET
        for original, back in zip(codes, snac._unflatten_codes(flat), strict=True):
            assert torch.equal(original, back)

    def test_unflattened_encode_output(self, snac):
        codes = snac.encode(torch.randn(4096) * 0.1, flatten=False)
        assert [c.shape[1] for c in codes] == [2, 4, 8]

    def test_roundtrip_length(self, snac):
        n = 2048 * 5
        tokens = snac.encode(torch.randn(n) * 0.1)
        audio = snac.decode(tokens)
        assert audio.dim() == 1
        assert audio.shape[0] == n

    def test_decode_accepts_1d_tokens_without_offset(self, snac):
        tokens = snac.encode(torch.randn(4096) * 0.1)[0] - SNACWrapper.AUDIO_TOKEN_OFFSET
        assert snac.decode(tokens).shape == (4096,)

    def test_decode_batch(self, snac):
        tokens = snac.encode(torch.randn(2, 4096) * 0.1)
        assert snac.decode(tokens).shape == (2, 4096)

    def test_decode_unflattened(self, snac):
        codes = snac.encode(torch.randn(4096) * 0.1, flatten=False)
        assert snac.decode(codes, is_flattened=False).shape == (4096,)
