"""Turn a folder of FLAC songs into YuE2 NAR training pairs.

For every song the public AR model generates semantic codec tokens from a text
request (style + lyrics, ``cot="off"``), and the public VAE encoder computes the
acoustic latents of the matching audio slice. The result is the same
(tokens, latents) pair the acoustic flow-matching path predicts at inference,
saved without any further model access.

Sidecar files: ``<song>.txt`` or ``<song>.lyrics.txt`` for lyrics,
``<song>.style.txt`` for a per-song style tag. Missing files fall back to the
command-line defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from yue2 import YuE2Pipeline
from yue2.protocol import Sampling as SemanticSampling, SongRequest
from yue2.storage import resolve_model

FPS = 25  # 48000 Hz / 1920 VAE downsample ratio
DOWNSAMPLE = 1920
CHUNK_SECONDS = 30  # VAE encode chunk size (avoids OOM on long audio)
PADDING_SECONDS = 2.0  # overlap padding per chunk side (latent-receptive-field safety)


def encode_chunked(vae, audio, sample_rate=48000, downsample=DOWNSAMPLE,
                   chunk_seconds=CHUNK_SECONDS, padding_seconds=PADDING_SECONDS):
    """Encode long audio in overlapping chunks to avoid VAE OOM."""
    total_samples = audio.shape[-1]
    chunk_samples = int(chunk_seconds * sample_rate)
    pad_samples = int(padding_seconds * sample_rate)

    if total_samples <= chunk_samples:
        with torch.inference_mode():
            return vae.encode(audio.to("cuda"))

    results, start = [], 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        enc_start = max(0, start - pad_samples)
        enc_end = min(total_samples, end + pad_samples)
        chunk = audio[:, :, enc_start:enc_end]
        with torch.inference_mode():
            lat = vae.encode(chunk.to("cuda")).cpu()
        trim_left = (start - enc_start) // downsample
        trim_right = lat.shape[-1] - (enc_end - end) // downsample
        results.append(lat[:, :, trim_left:trim_right])
        start = end
    return torch.cat(results, dim=-1)


def _identifier(name, seed):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "song"
    return f"{safe}-s{seed}"


def _sidecar(song, *suffixes):
    for suffix in suffixes:
        candidate = song.with_name(song.stem + suffix)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def load_audio(path):
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    samples, channels = data.shape
    if channels == 1:
        data = np.repeat(data, 2, axis=1)
    elif channels > 2:
        data = data[:, :2]
    tensor = torch.from_numpy(data).T.unsqueeze(0)  # [1, 2, S]
    if rate != 48000:
        length = max(int(round(samples * 48000 / rate)), 1)
        tensor = F.interpolate(tensor, size=length, mode="linear", align_corners=False)
    return tensor.contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("songs", type=Path, help="folder containing *.flac files")
    parser.add_argument("--out", type=Path, default=Path("train/dataset"))
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--style", default="Unknown genre, professional studio recording")
    parser.add_argument("--lyrics", default="[No lyrics text provided]")
    parser.add_argument("--clip-seconds", type=float, default=30.0,
                        help="0 = use full track length")
    parser.add_argument("--no-chunked", action="store_true",
                        help="disable chunked VAE encoding (risks OOM on long audio)")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--no-low-vram", action="store_true")
    parser.add_argument("--max-songs", type=int, default=0)
    args = parser.parse_args()

    songs = sorted(path for path in Path(args.songs).iterdir()
                   if path.is_file() and path.suffix.lower() == ".flac")
    if not songs:
        parser.error("no *.flac files found")
    if args.max_songs:
        songs = songs[: args.max_songs]
    if args.clip_seconds < 0:
        parser.error("clip-seconds must be non-negative (0 = full track)")
    seeds = [int(item) for item in args.seeds.replace(",", " ").split() if item.strip()]
    if not seeds:
        parser.error("seeds must be integers, e.g. 1,3,7")
    if not 0 <= args.cfg_scale <= 20:
        parser.error("cfg-scale must be in [0, 20]")

    out = Path(args.out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    resolve_model(args.model)
    resolve_model(args.vae)

    # Phase A: semantic codec tokens per (song, seed) from the public AR model.
    pending = []
    with YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda",
                                      low_vram=not args.no_low_vram, progress=False) as pipe:
        for song in songs:
            info = sf.info(str(song))
            frames = max(int(round(info.frames / info.samplerate * FPS)), 0)
            if args.clip_seconds == 0:
                target = frames
            else:
                target = min(frames, int(round(args.clip_seconds * FPS)))
            if target < 1:
                print(f"skip {song.name}: shorter than one latent frame")
                continue
            # Context window caps semantic sequence at 24576 total; cap frames
            # so the invariant len(prefix) + target + margin <= CONTEXT holds.
            target = min(target, 24576 - 512)
            style = _sidecar(song, ".style.txt") or args.style
            lyrics = _sidecar(song, ".lyrics.txt", ".txt") or args.lyrics
            for seed in seeds:
                request = SongRequest(style=style, lyrics=lyrics, cot="off",
                                      seed=seed, cfg_scale=args.cfg_scale)
                plan = pipe.plan(request=request)
                sampling = SemanticSampling(
                    temperature=1.0, top_p=0.95, top_k=100,
                    repetition_penalty=1.2, penalty_window=50,
                    min_tokens=min(32, target + 32), max_tokens=target + 32)
                semantic = pipe.generate_semantic(plan, sampling=sampling)
                codec = list(semantic.tokens)
                if not codec:
                    print(f"skip {song.name} seed {seed}: no codec tokens")
                    continue
                row = _identifier(song.stem, seed)
                np.save(data_dir / f"{row}.tokens.npy", np.asarray(codec, dtype=np.int64))
                np.save(data_dir / f"{row}.prefix.npy", np.asarray(plan.prefix, dtype=np.int64))
                pending.append({"id": row, "song": str(song), "style": style,
                                "lyrics": lyrics, "seed": seed,
                                "frames_avail": frames, "codec": len(codec)})
                print(f"{song.name} seed {seed}: {len(codec)} codec tokens")

    # Phase B: encode the matching audio slice with the public VAE encoder.
    from yue2.modeling_vae import YuE2VAE
    vae = YuE2VAE.from_pretrained(resolve_model(args.vae), decoder_only=False,
                                  device="cuda")
    entries, removed, latent_cache = [], [], {}
    try:
        for item in pending:
            row = item["id"]
            codec = np.load(data_dir / f"{row}.tokens.npy").tolist()
            frames = min(len(codec), item["frames_avail"])
            if frames < 1:
                removed.append(row)
                continue
            audio_key = item["song"]
            need = frames * DOWNSAMPLE
            if audio_key in latent_cache:
                cached_lat = latent_cache[audio_key]
                audio_len = cached_lat.shape[-1]
                if audio_len < need:
                    frames = audio_len // DOWNSAMPLE
                    if frames < 1:
                        removed.append(row)
                        continue
                latents = cached_lat
            else:
                audio = load_audio(item["song"])
                if audio.shape[-1] < need:
                    frames = audio.shape[-1] // DOWNSAMPLE
                    if frames < 1:
                        removed.append(row)
                        continue
                    need = frames * DOWNSAMPLE
                with torch.inference_mode():
                    if args.no_chunked:
                        latents = vae.encode(audio[:, :, :need].to("cuda")).cpu()[0]
                    else:
                        latents = encode_chunked(vae, audio[:, :, :need]).cpu()[0]
                latent_cache[audio_key] = latents
            frames = min(frames, latents.shape[-1])
            if frames < 1:
                removed.append(row)
                continue
            codec = codec[:frames]
            np.save(data_dir / f"{row}.tokens.npy", np.asarray(codec, dtype=np.int64))
            np.save(data_dir / f"{row}.latents.npy",
                    latents[:, :frames].t().contiguous().numpy().astype(np.float32))
            entries.append({**item, "frames": frames})
            print(f"{row}: {frames} latent frames -> {frames / FPS:.1f}s")
    finally:
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for row in removed:
        for suffix in (".tokens.npy", ".prefix.npy", ".latents.npy"):
            (data_dir / f"{row}{suffix}").unlink(missing_ok=True)

    (out / "manifest.json").write_text(
        json.dumps({"entries": entries, "fps": FPS, "downsample": 1920,
                    "sample_rate": 48000}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"dataset: {out} ({len(entries)} pairs)")


if __name__ == "__main__":
    main()