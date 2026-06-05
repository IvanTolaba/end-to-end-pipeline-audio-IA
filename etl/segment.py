import logging
import librosa#
from pathlib import Path
from typing import Dict, List
from config.settings import SAMPLE_RATE, WINDOW_TIME, OVERLAP

logger = logging.getLogger(__name__)

'''
def segment_audio_file(
    audio_path: str, 
    disease_name: str, 
    window_sec: float = 3.0, 
    overlap: float = 0.25
) -> List[Dict]:
'''

def segment_audio_file(
    audio_path: str, 
    disease_name: str
) -> List[Dict]:


    """
    Loads a full WAV file, checks sample rate, and slices it into segments 
    with a configured overlapping sliding window strategy.

    Args:
        audio_path: Path to the target audio file.
        disease_name: Label category name of the disease.
        window_sec: Slicing window length in seconds.
        overlap: Percent of overlapping between consecutive windows.
    Returns:
        List of dictionaries containing segment signals and metadata details.
        {
                "file_id": file_id,                 #audio name
                "segment_idx": segment_index,       #segment number
                "signal": segment_signal.tolist()  
            }
    """
    segments_metadata = []
    try:
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Target file not found: {audio_path}")

        # Load complete audio and resample to fixed rate
        y, sr = librosa.load(str(path_obj), sr=SAMPLE_RATE)

        # Fail-safe check for low-quality recordings
        # sr = sampling frequency
        if sr < 8000:
            logger.warning("Low sample rate (%s Hz) detected in: %s. Skipping.", sr, audio_path)
            return []

        # Convert temporal metadata to sample steps
        frame_samples = int(WINDOW_TIME * SAMPLE_RATE)
        hop_samples = int(WINDOW_TIME * SAMPLE_RATE * (1.0 - OVERLAP))
        file_id = f"{disease_name}_{path_obj.name}"

        segment_index = 0
        # Slide window across raw amplitude array
        for start in range(0, len(y) - frame_samples + 1, hop_samples):
            segment_signal = y[start:start + frame_samples]
            
            segments_metadata.append({
                "file_id": file_id,
                "segment_idx": segment_index,
                "signal": segment_signal.tolist()  # Convert numpy array to serializable Python list
            })
            segment_index += 1

    except Exception as error:
        logger.exception("Error slicing raw audio track %s: %s", audio_path, error)
        return []

    logger.info(
    "Generated %s segments from %s",
    len(segments_metadata),
    audio_path
    )

    return segments_metadata



