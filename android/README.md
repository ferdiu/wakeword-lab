# Android starter module

This module is intentionally a starter, not a production-ready detector. It contains the PCM ring buffer and capture cadence configuration. The microphone capture loop, Android log-mel parity, LiteRT model adapter, quantized tensor scale/zero-point handling, VAD, foreground service and lifecycle-safe callback API are not implemented yet.

Build prerequisites: JDK 17, Gradle 8.7+, Android SDK 35. Request `RECORD_AUDIO` at runtime. Persistent microphone capture on modern Android requires a user-visible, correctly declared microphone foreground service and is subject to OS restrictions. Do not silently start listening in the background.
