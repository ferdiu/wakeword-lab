from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import yaml


def _project_init(project: Path, phrase: str):
    for rel in ("data/raw/positive", "data/raw/negative", "data/raw/noise", "data/processed", "artifacts/experiments"):
        (project / rel).mkdir(parents=True, exist_ok=True)
    (project / "wakeword.json").write_text(json.dumps({"wake_word": phrase}, indent=2), encoding="utf-8")
    default = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    if default.exists():
        shutil.copy2(default, project / "wakeword.yaml")
    else:
        from .config import DEFAULT_CONFIG
        (project / "wakeword.yaml").write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    metadata = project / "data/raw/metadata.csv"
    if not metadata.exists():
        metadata.write_text("path,speaker_id,group_id,split,locale,phrase_start_s,phrase_end_s\n", encoding="utf-8")
    (project / "data/raw/README.md").write_text(
        "Put one-second or phrase-length WAV recordings in positive/ and negative/; common sample "
        "rates and mono/stereo files are converted to 16 kHz mono on ingest. Put ambient-only "
        "WAV files in noise/. Use metadata.csv to group clips by speaker_id "
        "and optionally pin a speaker/session group to train, validation or test.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(prog="wakeword-lab", description="Train and package tiny offline wake-word models")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("project"); init.add_argument("--wake-word", default="Hey Nova")
    prep = sub.add_parser("prepare"); prep.add_argument("project"); prep.add_argument("--config"); prep.add_argument("--seed", type=int); prep.add_argument("--no-augmentation", action="store_true")
    synth = sub.add_parser("synthesize"); synth.add_argument("project"); synth.add_argument("--phrase", required=True); synth.add_argument("--variants-per-locale", type=int, default=24); synth.add_argument("--seed", type=int, default=42); synth.add_argument("--locales", help="comma-separated locale override (e.g. it-IT); default is the built-in en-IN/en-US/en-GB/en-AU weighted pools")
    synth_neg = sub.add_parser("synthesize-negatives"); synth_neg.add_argument("project"); synth_neg.add_argument("--phrase", action="append"); synth_neg.add_argument("--phrases-file"); synth_neg.add_argument("--variants-per-locale", type=int, default=24); synth_neg.add_argument("--seed", type=int, default=42)
    record = sub.add_parser("record"); record.add_argument("project"); record.add_argument("--kind", choices=["positive", "negative", "noise"], default="positive"); record.add_argument("--speaker"); record.add_argument("--split", choices=["train", "validation", "test"]); record.add_argument("--sample-rate", type=int, default=16000); record.add_argument("--count", type=int)
    build_rec = sub.add_parser("build-recorder"); build_rec.add_argument("project"); build_rec.add_argument("--install", action="store_true")
    fetch_rec = sub.add_parser("fetch-recordings"); fetch_rec.add_argument("project"); fetch_rec.add_argument("--speaker", default="phone"); fetch_rec.add_argument("--delete", action="store_true")
    fetch = sub.add_parser("fetch-negatives"); fetch.add_argument("project"); fetch.add_argument("--count", type=int, default=None); fetch.add_argument("--seed", type=int, default=42); fetch.add_argument("--source", choices=["speech-commands-mini", "speech-commands-full", "librispeech", "librispeech-train", "musan-speech", "musan-music", "esc-50"], default="speech-commands-mini")
    fetch_ns = sub.add_parser("fetch-noise"); fetch_ns.add_argument("project"); fetch_ns.add_argument("--environments", default="DKITCHEN,OOFFICE,STRAFFIC"); fetch_ns.add_argument("--seed", type=int, default=42)
    noise = sub.add_parser("generate-noise"); noise.add_argument("project"); noise.add_argument("--count", type=int, default=400); noise.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train"); train.add_argument("project"); train.add_argument("--run-name", default="default"); train.add_argument("--config"); train.add_argument("--epochs", type=int); train.add_argument("--batch-size", type=int); train.add_argument("--width", type=int); train.add_argument("--dropout", type=float)
    export = sub.add_parser("export"); export.add_argument("project"); export.add_argument("--run-name", default="default"); export.add_argument("--config")
    evaluate = sub.add_parser("evaluate"); evaluate.add_argument("project"); evaluate.add_argument("--run-name", default="default")
    experiment = sub.add_parser("experiment"); experiment.add_argument("project"); experiment.add_argument("--run-name", default="default"); experiment.add_argument("--config"); experiment.add_argument("--width", type=int); experiment.add_argument("--dropout", type=float)
    sweep = sub.add_parser("sweep"); sweep.add_argument("project"); sweep.add_argument("--widths", default="8,12,16,24"); sweep.add_argument("--dropouts", default="0.1,0.15,0.2"); sweep.add_argument("--config")
    stream = sub.add_parser("evaluate-stream"); stream.add_argument("project"); stream.add_argument("recordings_dir"); stream.add_argument("--run-name", default="default"); stream.add_argument("--annotations"); stream.add_argument("--vad-rms", type=float, default=0.008); stream.add_argument("--cooldown-ms", type=int, default=1500); stream.add_argument("--merge-gap", type=float, default=1.0); stream.add_argument("--threshold", type=float)
    package = sub.add_parser("package-android"); package.add_argument("project"); package.add_argument("--run-name", default="default"); package.add_argument("--destination")
    report = sub.add_parser("report"); report.add_argument("project"); report.add_argument("--run-name")
    compare = sub.add_parser("compare"); compare.add_argument("project")

    args = parser.parse_args()
    project = Path(getattr(args, "project", "."))
    if args.command == "init":
        _project_init(Path(args.project), args.wake_word)
    elif args.command == "prepare":
        from .dataset import prepare
        print(json.dumps(prepare(project, seed=args.seed, config_path=args.config, augment=not args.no_augmentation)["summary"], indent=2))
    elif args.command == "synthesize":
        from .synthesize import synthesize
        import asyncio
        locales = [x.strip() for x in args.locales.split(",") if x.strip()] if args.locales else None
        print(f"Generated {asyncio.run(synthesize(project, args.phrase, args.variants_per_locale, args.seed, locales))} WAV files")
    elif args.command == "synthesize-negatives":
        import asyncio

        from .synthesize import _read_phrases_file, synthesize_negatives
        phrases = list(args.phrase or [])
        if args.phrases_file:
            phrases += _read_phrases_file(args.phrases_file)
        if not phrases:
            raise SystemExit("synthesize-negatives needs at least one --phrase or --phrases-file")
        print(f"Generated {asyncio.run(synthesize_negatives(project, phrases, args.variants_per_locale, args.seed))} WAV files")
    elif args.command == "record":
        import os

        from .record import run_recording_session
        speaker = args.speaker or os.environ.get("USER") or "unknown"
        run_recording_session(project, kind=args.kind, speaker=speaker, split=args.split,
                              sample_rate=args.sample_rate, count=args.count)
    elif args.command == "build-recorder":
        from .recorder import build_recorder
        print(build_recorder(project, install=args.install))
    elif args.command == "fetch-recordings":
        from .recorder import fetch_recordings
        print(json.dumps(fetch_recordings(project, speaker=args.speaker, delete=args.delete), indent=2))
    elif args.command == "fetch-negatives":
        from .fetch_negatives import fetch_negatives

        fetch_negatives(project, count=args.count, seed=args.seed, source=args.source)
    elif args.command == "fetch-noise":
        from .fetch_noise import fetch_noise
        environments = [x.strip() for x in args.environments.split(",") if x.strip()]
        fetch_noise(project, environments=environments, seed=args.seed)
    elif args.command == "generate-noise":
        from .noise import generate_noise
        generated = generate_noise(project, count=args.count, seed=args.seed)
        print(f"Generated {len(generated)} one-second noise WAV files in {project / 'data/raw/noise'}")
    elif args.command == "train":
        from .train import train as run_train
        run_train(project, epochs=args.epochs, batch_size=args.batch_size, run_name=args.run_name,
                  config_path=args.config, width=args.width, dropout=args.dropout)
    elif args.command == "export":
        from .export import export as run_export
        print(json.dumps(run_export(project, args.run_name, args.config), indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate_test
        print(json.dumps(evaluate_test(project, args.run_name), indent=2))
    elif args.command == "experiment":
        from .experiment import run_experiment
        print(json.dumps(run_experiment(project, args.run_name, args.config, width=args.width, dropout=args.dropout), indent=2))
    elif args.command == "sweep":
        from .experiment import sweep as run_sweep
        widths = [int(x) for x in args.widths.split(",") if x.strip()]
        dropouts = [float(x) for x in args.dropouts.split(",") if x.strip()]
        print(json.dumps(run_sweep(project, widths, dropouts, args.config), indent=2))
    elif args.command == "evaluate-stream":
        from .stream import evaluate_stream
        print(json.dumps(evaluate_stream(project, args.recordings_dir, args.run_name, args.annotations, args.vad_rms, args.cooldown_ms, args.merge_gap, args.threshold), indent=2))
    elif args.command == "package-android":
        art = project / "artifacts/experiments" / args.run_name
        dest = Path(args.destination) if args.destination else Path(__file__).resolve().parents[2] / "android/demo/src/main/assets"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(art / "wakeword_int8.tflite", dest / "wakeword_int8.tflite")
        shutil.copy2(art / "model_metadata.json", dest / "model_metadata.json")
        print(f"Packaged model + metadata in {dest}")
    elif args.command == "report":
        from .reporting import generate_report
        if args.run_name:
            print(f"Report written to {generate_report(project, args.run_name)}")
        else:
            for file in sorted((project / "artifacts/experiments").glob("*/summary.json")):
                print(f"## {file.parent.name}\n{file.read_text(encoding='utf-8')}")
                try:
                    generate_report(project, file.parent.name)
                except Exception as exc:
                    print(f"warning: report refresh failed for {file.parent.name}: {exc}")
    elif args.command == "compare":
        import csv
        path = project / "artifacts/sweep_results.csv"
        if not path.exists():
            print("No sweep results found. Run `wakeword-lab sweep` first.")
        else:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows.sort(key=lambda row: (
                int(row["validation_false_positives"] or 0),
                -float(row["validation_recall"] or 0),
                int(row["model_size_bytes"] or 0),
            ))
            print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
