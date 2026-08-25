"""Data model and loader tests."""

from __future__ import annotations

import numpy as np
import pytest

from statorscope import Motor, Recording


class TestRecording:
    def test_shape_and_properties(self):
        rec = Recording.from_array(np.zeros((1000, 3)), fs=500.0, name="x")
        assert len(rec) == 1000
        assert rec.n_phases == 3
        assert rec.duration_s == pytest.approx(2.0)
        assert rec.freq_resolution_hz == pytest.approx(0.5)

    def test_single_phase(self):
        rec = Recording.from_array(np.zeros(500), fs=100.0)
        assert rec.n_phases == 1
        assert rec.phase(0).shape == (500,)

    def test_phase_by_name(self):
        data = np.column_stack([np.full(100, 1.0), np.full(100, 2.0), np.full(100, 3.0)])
        rec = Recording.from_array(data, fs=100.0)
        # mean is removed, so a constant column becomes zero
        assert np.allclose(rec.phase("S"), 0.0)
        assert np.allclose(rec.phase(1), rec.phase("s"))

    def test_phase_removes_mean(self):
        rec = Recording.from_array(np.full((100, 1), 615.0), fs=100.0)
        assert np.allclose(rec.phase(0), 0.0)

    def test_bad_phase_name(self):
        rec = Recording.from_array(np.zeros((10, 3)), fs=10.0)
        with pytest.raises(KeyError, match="phase must be"):
            rec.phase("Q")

    def test_rejects_bad_fs(self):
        with pytest.raises(ValueError, match="fs"):
            Recording.from_array(np.zeros(100), fs=0.0)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            Recording(np.zeros((2, 2, 2)), fs=10.0)

    def test_rejects_mismatched_timestamps(self):
        with pytest.raises(ValueError, match="timestamps length"):
            Recording(np.zeros(100), fs=10.0, timestamps=np.zeros(50))

    def test_with_fs_is_a_copy(self):
        rec = Recording.from_array(np.zeros(100), fs=100.0)
        other = rec.with_fs(200.0)
        assert rec.fs == 100.0
        assert other.fs == 200.0


class TestFromText:
    def test_reads_timestamped_log(self, tmp_path):
        p = tmp_path / "cap.txt"
        rows = [f"{i} {600 + i % 7} {610 + i % 5} {620 + i % 3}" for i in range(0, 2000, 2)]
        p.write_text("\n".join(rows))
        rec = Recording.from_text(p, time_column=0, time_unit="ms")
        assert rec.n_phases == 3
        assert rec.timestamps is not None
        assert rec.fs == pytest.approx(500.0, rel=0.01)
        assert rec.name == "cap"

    def test_without_time_column_requires_fs(self, tmp_path):
        p = tmp_path / "raw.txt"
        p.write_text("\n".join("1 2 3" for _ in range(100)))
        with pytest.raises(ValueError, match="fs is required"):
            Recording.from_text(p, time_column=None)

    def test_without_time_column_uses_supplied_fs(self, tmp_path):
        p = tmp_path / "raw.txt"
        p.write_text("\n".join("1 2 3" for _ in range(100)))
        rec = Recording.from_text(p, time_column=None, fs=1000.0)
        assert rec.fs == 1000.0
        assert rec.n_phases == 3

    def test_rejects_non_increasing_time(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("\n".join("5 1 2 3" for _ in range(50)))
        with pytest.raises(ValueError, match="not increasing"):
            Recording.from_text(p, time_column=0)

    def test_selects_explicit_columns(self, tmp_path):
        p = tmp_path / "wide.txt"
        p.write_text("\n".join(f"{i} 1 2 3 4 5" for i in range(0, 200, 2)))
        rec = Recording.from_text(p, time_column=0, current_columns=(1, 3))
        assert rec.n_phases == 2


class TestMotorValidation:
    def test_rejects_bad_rotor_bars(self):
        with pytest.raises(ValueError, match="rotor_bars"):
            Motor(pole_pairs=2, rotor_bars=0)

    def test_rejects_bad_line_hz(self):
        with pytest.raises(ValueError, match="line_hz"):
            Motor(pole_pairs=2, line_hz=0.0)

    def test_is_hashable_and_frozen(self):
        m = Motor(pole_pairs=2)
        assert hash(m)
        with pytest.raises(AttributeError):
            m.pole_pairs = 4  # type: ignore[misc]
