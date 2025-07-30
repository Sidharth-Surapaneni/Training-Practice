from datasets import load_dataset
from scipy.signal import resample
import numpy as np
from tqdm import tqdm
import re
import json
from pathlib import Path
import unicodedata

def resample_audio(audio_array, source_sr, target_sr=16000):
    """Resample audio array to target sampling rate.
    
    Args:
        audio_array (np.ndarray): Audio data as numpy array
        source_sr (int): Source sampling rate
        target_sr (int): Target sampling rate, defaults to 16000
        
    Returns:
        tuple: (resampled_audio, target_sr)
    """
    if source_sr != target_sr:
        target_length = int(target_sr * len(audio_array) / source_sr)
        return resample(audio_array, target_length), target_sr
    return audio_array, target_sr


def extract_audio(example, target_sr=16000):
    """Extract and resample audio from an example.
    
    Args:
        example (dict): Dictionary containing audio data
        target_sr (int): Target sampling rate for audio resampling
        
    Returns:
        tuple: (processed_audio, audio_length_seconds, original_sr) or (None, 0, 0) if extraction fails
              processed_audio: resampled audio array
              audio_length_seconds: length of the audio in seconds
              original_sr: original sampling rate before resampling
    """
    try:
        # Extract audio information without creating copies
        audio_array = None
        sampling_rate = None
        
        if "audio" in example:
            audio_array = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
        elif "context" in example:
            audio_array = example["context"]["array"]
            sampling_rate = example["context"]["sampling_rate"]
        else:
            return None, 0, 0
            
        # Calculate audio length before resampling
        audio_length_seconds = len(audio_array) / sampling_rate
            
        # Resample audio if needed
        if sampling_rate != target_sr:
            processed_audio, _ = resample_audio(audio_array, sampling_rate, target_sr)
        else:
            processed_audio = audio_array
            
        return processed_audio, audio_length_seconds, sampling_rate
        
    except Exception as e:
        # Return None if processing fails
        return None, 0, 0


class TextProcessor:
    """Class to handle text processing operations including Unicode normalization and character encoding."""
    
    def __init__(self, normalization_form='NFKC', allowed_chars=None, lowercase=True):
        """Initialize the TextProcessor.
        
        Args:
            normalization_form (str): Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
            allowed_chars (str or None): Regex pattern of allowed characters, defaults to alphanumeric and spaces
            lowercase (bool): Whether to convert text to lowercase
        """
        self.normalization_form = normalization_form
        self.allowed_chars = allowed_chars or r'[^a-zA-Z0-9\s]'
        self.lowercase = lowercase
        
    def normalize_unicode(self, text):
        """Normalize Unicode characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Unicode normalized text
        """
        if text is None:
            return None
        return unicodedata.normalize(self.normalization_form, text)
    
    def filter_chars(self, text):
        """Filter out characters not matching the allowed pattern.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with only allowed characters
        """
        if text is None:
            return None
        return re.sub(self.allowed_chars, '', text)
    
    def process(self, text):
        """Process text through the full pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Fully processed text
        """
        if text is None:
            return None
        
        # Apply processing pipeline
        result = text
        result = self.normalize_unicode(result)
        if self.lowercase:
            result = result.lower()
        result = self.filter_chars(result)
        return result


def process_text(example, text_processor=None):
    """Process text from an example using configurable text processor.
    
    Args:
        example (dict): Dictionary containing text data
        text_processor (TextProcessor, optional): Text processor instance
        
    Returns:
        str or None: Processed text, or None if no text is available
    """
    try:
        # Use default text processor if none provided
        if text_processor is None:
            text_processor = TextProcessor()
            
        # Check for text in different possible field names
        if "sentence" in example:
            original_text = example["sentence"]
        elif "text" in example:
            original_text = example["text"]
        elif "normalized_text" in example:
            original_text = example["normalized_text"]
        else:
            return None
            
        # Process text through the pipeline
        return text_processor.process(original_text)
    except Exception:
        return None




def process_audio(output_dir="./data_stats", text_processor=None):
    """Process audio datasets and log statistics.
    
    Args:
        output_dir (str): Directory to save the dataset statistics
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stats_file = Path(output_dir) / "dataset_stats.jsonl"
    
    # Dictionary to hold all datasets
    datasets = {}
    
    # Load Common Voice datasets
    cv_langs = ["en", "es", "hi", "ta", "sw", "am", "as"]
    for lang_code in cv_langs:
        dataset_name = f"cv_{lang_code}"
        try:
            datasets[dataset_name] = load_dataset("mozilla-foundation/common_voice_17_0", lang_code, split="train", trust_remote_code=True)
            print(f"Loaded Common Voice dataset: {lang_code}")
        except Exception as e:
            print(f"Failed to load Common Voice dataset {lang_code}: {str(e)}")
    
    # Load LibriSpeech dataset
    try:
        datasets["librispeech"] = load_dataset("openslr/librispeech_asr", "all", split="train.other.500", trust_remote_code=True)
        print("Loaded LibriSpeech dataset: train.other.500")
    except Exception as e:
        print(f"Failed to load LibriSpeech dataset: {str(e)}")
        
    # Load VoxPopuli datasets for English and Spanish
    vp_langs = ["en", "es"]
    for lang_code in vp_langs:
        dataset_name = f"voxpopuli_{lang_code}"
        try:
            datasets[dataset_name] = load_dataset("facebook/voxpopuli", lang_code, split="train", trust_remote_code=True)
            print(f"Loaded VoxPopuli dataset: {lang_code}")
        except Exception as e:
            print(f"Failed to load VoxPopuli dataset {lang_code}: {str(e)}")
    
    # Process and resample all audio to 16000 Hz
    resampled_audio = []
    text_data = []  # Initialize text data list once
    
    # Prepare for dataset statistics
    dataset_stats = []
    
    # Process each dataset
    for dataset_id, dataset in datasets.items():
        # Track dataset statistics
        total_audio_length = 0.0
        successful_samples = 0
        failed_samples = 0
        unique_speakers = set()  # Set to track unique client_ids/speaker_ids
        gender_count = {"male": 0, "female": 0, "other": 0}  # Count gender distribution
        
        # Process each example with progress bar
        for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_id} audio"):
            try:
                # Get example and extract audio efficiently
                example = dataset[i]
                
                # Extract audio and get length in one step
                processed_audio, audio_length, original_sr = extract_audio(example)
                processed_text = process_text(example, text_processor)
                
                # Skip if audio processing failed
                if processed_audio is None:
                    failed_samples += 1
                    continue
                    
                # Update statistics
                total_audio_length += audio_length
                successful_samples += 1
                
                # Track unique speakers by client_id or speaker_id
                if "client_id" in example:
                    unique_speakers.add(example["client_id"])
                elif "speaker_id" in example:
                    unique_speakers.add(example["speaker_id"])
                    
                # Track gender distribution if available
                if "gender" in example:
                    gender = example["gender"].lower() if example["gender"] else "other"
                    if gender in ["m", "male"]:
                        gender_count["male"] += 1
                    elif gender in ["f", "female"]:
                        gender_count["female"] += 1
                    else:
                        gender_count["other"] += 1
                    
                resampled_audio.append(processed_audio)
                if processed_text is not None:
                    text_data.append(processed_text)
                del processed_audio
                del processed_text
            except Exception as e:
                # Skip problematic entries
                failed_samples += 1
                continue
                
        # Log dataset statistics
        hours = total_audio_length / 3600
        # Calculate gender percentages
        total_with_gender = sum(gender_count.values())
        gender_percent = {}
        if total_with_gender > 0:
            for gender, count in gender_count.items():
                gender_percent[f"{gender}_percent"] = round((count / total_with_gender) * 100, 2)
        
        lang_stats = {
            "dataset_id": dataset_id,
            "total_samples": len(dataset),
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "total_audio_length_seconds": total_audio_length,
            "total_audio_length_hours": hours,
            "average_sample_length": total_audio_length / successful_samples if successful_samples > 0 else 0,
            "unique_speakers": len(unique_speakers),  # Number of unique speakers
            "gender_distribution": gender_count,
            "gender_percentage": gender_percent
        }
        dataset_stats.append(lang_stats)
        
        # Write to JSONL file as we go
        with open(stats_file, "a") as f:
            f.write(json.dumps(lang_stats) + "\n")
            
        # Prepare gender distribution string
        gender_str = ""
        if total_with_gender > 0:
            gender_str = f", Gender: {gender_count['male']} male ({gender_percent['male_percent']}%), {gender_count['female']} female ({gender_percent['female_percent']}%)"
            
        print(f"{dataset_id} dataset: {successful_samples} samples, {hours:.2f} hours of audio, {len(unique_speakers)} unique speakers{gender_str}")

    # Return both audio and text data for further processing
    return resampled_audio, text_data, dataset_stats