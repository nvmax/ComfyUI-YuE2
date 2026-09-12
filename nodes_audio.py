"""Audio Save and Preview Node for YuE2 in ComfyUI."""

import os
import time
import numpy as np
import soundfile as sf
import torch
import folder_paths

class YuE2AudioSavePreview:
    """Saves generated songs and renders an interactive audio playback widget in the ComfyUI interface."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "YuE2/song"}),
                "save_mode": (["Save to Output", "Preview Only (Temp)"], {"default": "Save to Output"}),
                "format": (["flac", "wav", "mp3"], {"default": "mp3"}),
            },
            "optional": {
                "mp3_bitrate": (["320k", "256k", "192k", "128k"], {"default": "320k"}),
                "bit_depth": (["PCM_24", "PCM_16", "FLOAT"], {"default": "PCM_24"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT")
    RETURN_NAMES = ("audio", "file_path", "duration_seconds")
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True
    CATEGORY = "YuE2/Audio"

    def save_and_preview(self, audio, filename_prefix="YuE2/song", save_mode="Save to Output", format="mp3", mp3_bitrate="320k", bit_depth="PCM_24", **kwargs):
        if not audio or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("YuE2AudioSavePreview: Received invalid or empty audio tensor dictionary.")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        is_temp = (save_mode == "Preview Only (Temp)")
        folder_type = "temp" if is_temp else "output"
        base_dir = folder_paths.get_temp_directory() if is_temp else folder_paths.get_output_directory()

        # Sanitize prefix and compute save location
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, base_dir
        )
        os.makedirs(full_output_folder, exist_ok=True)

        results_ui = []
        saved_paths = []
        total_duration = 0.0

        # Handle batches [batch, channels, samples]
        batch_size = waveform.shape[0]
        for b in range(batch_size):
            audio_track = waveform[b].detach().cpu().numpy()

            # Shape: [channels, samples] -> [samples, channels] for soundfile
            if audio_track.ndim == 2:
                audio_track = audio_track.T
            elif audio_track.ndim == 1:
                audio_track = audio_track[:, np.newaxis]

            # Calculate duration
            duration = audio_track.shape[0] / float(sample_rate)
            total_duration += duration

            file_stem = f"{filename}_{counter:05}" if batch_size == 1 else f"{filename}_{counter:05}_b{b+1:02}"
            file_name = f"{file_stem}.{format.lower()}"
            full_path = os.path.join(full_output_folder, file_name)

            if format.lower() == "mp3":
                try:
                    sf.write(full_path, audio_track, sample_rate, format="MP3")
                except Exception:
                    import torchaudio, warnings
                    track_tensor = waveform[b].detach().cpu().float()
                    if track_tensor.ndim == 1:
                        track_tensor = track_tensor.unsqueeze(0).repeat(2, 1)
                    elif track_tensor.shape[0] == 1:
                        track_tensor = track_tensor.repeat(2, 1)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        torchaudio.save(full_path, track_tensor, sample_rate)
            else:
                # Choose subtype based on format and bit depth
                subtype = None
                if format.lower() == "flac":
                    subtype = "PCM_24" if bit_depth == "PCM_24" else "PCM_16"
                elif format.lower() == "wav":
                    if bit_depth == "FLOAT":
                        subtype = "FLOAT"
                    elif bit_depth == "PCM_24":
                        subtype = "PCM_24"
                    else:
                        subtype = "PCM_16"
                sf.write(full_path, audio_track, sample_rate, subtype=subtype)

            results_ui.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": folder_type
            })
            saved_paths.append(full_path)
            counter += 1

        primary_path = saved_paths[0] if saved_paths else ""
        print(f"[YuE2 ComfyUI] Saved audio ({format.upper()}) to: {primary_path} ({total_duration:.1f}s)")

        return {
            "ui": {
                "audio": results_ui
            },
            "result": (audio, primary_path, total_duration)
        }
