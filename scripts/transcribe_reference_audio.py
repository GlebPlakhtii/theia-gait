"""Locate archival transcript anchors with local ASR; never assign truth labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio_directory', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-metadata', type=Path, required=True)
    args = parser.parse_args()
    wavs = sorted(args.audio_directory.glob('*.wav'))
    if not wavs:
        raise ValueError('No WAV inputs')
    if any(p.with_suffix('.whisper.json').exists() for p in wavs):
        raise ValueError('Refuse to overwrite an existing transcription')
    metadata = json.loads(args.model_metadata.read_text())
    model_sha = next(x['lfs']['sha256'] for x in metadata['siblings']
                     if x['rfilename'] == 'model.safetensors')
    if digest(args.model / 'model.safetensors') != model_sha:
        raise ValueError('Model hash mismatch')

    import numpy as np
    import torch
    from scipy.io import wavfile
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    torch.set_num_threads(4)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, local_files_only=True, use_safetensors=True,
        dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    asr = pipeline('automatic-speech-recognition', model=model,
                   tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor,
                   device='cuda:0', dtype=torch.float16)
    settings = dict(language='english', task='transcribe', num_beams=3,
                    condition_on_prev_tokens=False, temperature=0.0)
    for wav in wavs:
        sr, samples = wavfile.read(wav)
        if sr != 16000 or samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError('Expected mono 16 kHz signed-16-bit PCM')
        started = time.monotonic()
        with torch.inference_mode():
            result = asr({'array': samples.astype(np.float32) / 32768.0,
                          'sampling_rate': sr}, return_timestamps='word',
                         generate_kwargs=settings)
        record = dict(source_audio=str(wav.resolve()), source_sha256=digest(wav),
                      model='openai/whisper-large-v3', revision=metadata['sha'],
                      weight_sha256=model_sha, settings=settings,
                      runner_sha256=digest(Path(__file__)),
                      created_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_s=time.monotonic() - started, result=result,
                      role='Unverified machine transcription for archive alignment; not independent event or decision labels')
        wav.with_suffix('.whisper.json').write_text(json.dumps(record, indent=2) + '\n')
        print(wav.name, round(record['elapsed_s'], 2), 'seconds; locator saved', flush=True)


if __name__ == '__main__':
    main()
