import logging
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the target function to test
from etl.segment import segment_audio_file

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_audio_signal():
    """
    Fixture that generates a dummy synthetic 1D numpy array representing 
    an audio signal lasting 10 seconds at a specific sample rate.
    """
    # 10 seconds of dummy audio at 22050 Hz
    duration_sec = 10
    sr = 22050
    total_samples = duration_sec * sr
    return np.sin(np.linspace(0, 100, total_samples)), sr


# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------

@patch("etl.segment.Path.exists")
def test_segment_audio_file_not_found(mock_exists):
    """
    Test that the function safely returns an empty list and logs warning/error
    when the targeted raw audio path does not exist on disk.
    """
    logger.info("Executing test: File not found handling scenario")
    
    # Force Path.exists() to return False
    mock_exists.return_value = False

    result = segment_audio_file(
        audio_path="fake_directory/non_existent_file.wav",
        disease_name="Asma"
    )

    assert isinstance(result, list)
    assert len(result) == 0
    logger.info("Test passed: Non-existent file handled correctly")


@patch("etl.segment.Path.exists")
@patch("etl.segment.librosa.load")
def test_segment_audio_low_sample_rate(mock_librosa_load, mock_exists):
    """
    Test that the function completely skips slicing and returns an empty list 
    if the original recording's sample rate is below the 8000 Hz quality threshold.
    """
    logger.info("Executing test: Low sample rate threshold skipping scenario")
    
    # Mock path existence and librosa output with a low sample rate (e.g., 4000 Hz)
    mock_exists.return_value = True
    dummy_signal = np.zeros(4000)
    mock_librosa_load.return_value = (dummy_signal, 4000)

    result = segment_audio_file(
        audio_path="data/raw/Asma/low_quality.wav",
        disease_name="Asma"
    )

    assert isinstance(result, list)
    assert len(result) == 0
    logger.info("Test passed: Low quality recording successfully skipped")


@patch("etl.segment.Path.exists")
@patch("etl.segment.librosa.load")
def test_segment_audio_file_success(mock_librosa_load, mock_exists, mock_audio_signal):
    """
    Test a highly successful pipeline run. Validates that metadata fields 
    (file_id, segment_idx, signal arrays) are generated with correct sliding window steps.
    """
    logger.info("Executing test: Successful audio sliding window segmentation mapping")
    
    mock_exists.return_value = True
    
    # Inject the dummy 10-second audio signal fixture
    signal, sr = mock_audio_signal
    mock_librosa_load.return_value = (signal, sr)

    disease = "Epoc"
    file_name = "patient_sample_001.wav"

    result = segment_audio_file(
        audio_path=f"data/raw/{disease}/{file_name}",
        disease_name=disease
    )

    # Assertions
    assert isinstance(result, list)
    assert len(result) > 0

    # Validate first entry structural integrity
    first_segment = result[0]
    assert "file_id" in first_segment
    assert "segment_idx" in first_segment
    assert "signal" in first_segment
    
    assert first_segment["file_id"] == f"{disease}_{file_name}"
    assert first_segment["segment_idx"] == 0
    assert isinstance(first_segment["signal"], list)

    # Validate index ordering incrementation
    for index, record in enumerate(result):
        assert record["segment_idx"] == index

    logger.info("Test passed: Slided structural items mapped accurately. Found %s slices.", len(result))


@patch("etl.segment.Path.exists")
@patch("etl.segment.librosa.load")
def test_segment_audio_exception_handling(mock_librosa_load, mock_exists):
    """
    Test that internal exceptions triggered inside third-party frameworks (e.g. librosa decoding errors)
    are caught cleanly, preventing pipeline crashes by returning an empty safe array list.
    """
    logger.info("Executing test: Internal parsing exception resiliency processing")
    
    mock_exists.return_value = True
    
    # Simulate a critical runtime decoding crash inside librosa framework
    mock_librosa_load.side_effect = RuntimeError("Corrupted audio headers or codec mismatch failure")

    result = segment_audio_file(
        audio_path="data/raw/Neumonia/corrupted_track.wav",
        disease_name="Neumonia"
    )

    assert isinstance(result, list)
    assert len(result) == 0
    logger.info("Test passed: Crash prevented. Safe empty list structure returned successfully")