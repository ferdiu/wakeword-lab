# Rooted-device options: hiding (or eliminating) the mic privacy indicator

Background: on Android 12+, an app that continuously records audio shows the
mandatory green privacy indicator (AOSP "Privacy indicators" spec, CDD 2.2.5).
There is no per-app or per-API way for a normal app to suppress it. This note
collects the options that open up on **rooted devices**, plus a feasibility
analysis of the SoundTrigger (always-on DSP) path for this project's model.

These options are for personal/lab devices only — none of them are shippable in
a Play-installable app.

## Option 1 — Hide the indicator via an Xposed/LSPosed module (cosmetic)

The mic keeps streaming to the app exactly as today; a SystemUI hook suppresses
the indicator.

- **Location Indicator Whitelist** (Xposed module): despite the name, it
  disables the indicator **per selected application** for mic/camera/location —
  the closest thing to the "disable it for one app" toggle that doesn't exist
  in stock settings.
- **GreenDotHide** (LSPosed module repository, `com.dorian15.greendothide`):
  hides indicators globally with per-type toggles (mic/camera/location/media
  projection). Actively maintained; confirmed working on AOSP 12–16,
  LineageOS 23.2, MIUI/HyperOS.

Caveats: hooks into SystemUI are fragile across Android/OEM updates (a bad hook
can crash-loop SystemUI; the module documents a safe-mode recovery), and they
hide the indicator rather than removing the reason for it.

## Option 2 — Run detection on the low-power DSP via SoundTrigger (by design)

This is the only path that removes the indicator *legitimately*: detection runs
on the audio DSP, audio never enters the app's process, so there is no mic
attribution and no indicator — the same reason "Hey Google" shows nothing.

Root is what makes this reachable for a third-party app: the SoundTrigger API
requires the assistant/`VoiceInteractionService` role and signature-privileged
permissions (e.g. `CAPTURE_AUDIO_HOTWORD`), which root can grant via a
privileged-app install (`/system/priv-app` Magisk module) or Shizuku-as-root.

**Working precedent:** AmbientMusicMod ports Pixel's Now Playing to arbitrary
rooted devices over the same kind of privileged ambient-audio path (no mic
indicator), and documents the per-device pitfalls (`sound_trigger_platform_info.xml`
overlays, Google-app service bindings, etc.).

### Do DSPs have a common interface?

Partially — **at the Android API level yes, at the model level no**.

- The **SoundTrigger HAL (STHAL)** is a standardized, versioned AOSP interface
  (`hardware/interfaces/soundtrigger/`). Any app (with the right privileges)
  loads a sound model via `loadSoundModel()` / `loadPhraseSoundModel()` and
  receives discrete trigger events; the app then opens a normal `AudioRecord`
  around the trigger to capture actual audio. This layer is common across
  vendors.
- The **sound model itself is an opaque, vendor-defined blob**. The HAL
  contract only says "model + implementation-specific engine parameters".
  Each vendor's DSP stack (Qualcomm SVA / Listen engine on Hexagon aDSP/WDSP,
  MediaTek, Samsung, etc.) has its own model format, its own training/tooling
  pipeline, and its own keyword engine. There is no standard "upload a TFLite
  file to the DSP" interface.

### Can the DSP run a small neural network like ours?

Our model is a small depthwise-separable CNN (`wakeword_ds_cnn`: one 3x3 conv,
five DS blocks, global average pool, single sigmoid output) over a 98x40 log-mel
input, exported as int8 TFLite. In absolute terms this is tiny — well within
what these DSPs can compute.

The hardware is not the problem:

- Qualcomm Hexagon aDSP/WDSP exists precisely for low-power always-on audio,
  and modern SVA versions explicitly support **CNN-based keyword spotting** on
  the DSP as a second stage (phoneme-graph first stage, CNN second stage with
  user verification).
- TFLite-quantized models demonstrably run on Hexagon (the TFLite Hexagon
  delegate, NNAPI), and int8 depthwise CNNs are the canonical workload.

The **tooling and access** are the problem:

- Vendor keyword engines (e.g. Qualcomm SVA) consume models in proprietary
  formats produced by vendor/OEM tooling — historically phoneme-graph models
  trained through Qualcomm's pipeline, with CNN variants aimed at OEM partners.
  There is no public, supported converter from an arbitrary TFLite graph to an
  SVA sound model.
- Practically, getting a custom model onto the DSP means one of:
  - reverse-engineering / patching the vendor sound model format (per-device,
    brittle, undocumented), or
  - replacing the vendor engine entirely on the aDSP (requires Qualcomm Hexagon
    SDK access, signed firmware constraints, per-device work), or
  - working with the vendor/OEM (not realistic for an individual project).
- Everything is per-SoC and often per-device: different Hexagon generations,
  different firmware, different `sound_trigger_platform_info.xml` tuning.

**Verdict:** technically feasible — the DSP can trivially run this class of
model, and the SoundTrigger HAL gives a clean event interface — but it is a
large, device-specific reverse-engineering/integration effort with no portable
outcome. Realistic only as a research project on one specific rooted phone,
not as a feature of this library.

## Option 3 — Install as a privileged system app (middle ground)

Root allows pushing the APK to `/system/priv-app` and granting privileged
permissions. Mic usage attributed to the system is skipped by AOSP's
privacy-indicator logic, so the dot disappears. Simpler than SoundTrigger (no
model conversion — the app keeps running its own TFLite inference on the CPU),
but blunter: it exempts the app from the privacy model rather than moving
detection to the low-power path, so power cost stays high.

## Recommendation

- Personal/lab device, minimal effort: Option 1 (per-app hiding with Location
  Indicator Whitelist).
- Same goal, no Xposed fragility: Option 3.
- Research-grade, indicator-free *and* low-power: Option 2, scoped to one
  device, expecting vendor-format reverse engineering. The model size is a
  non-issue; the vendor toolchain is the whole battle.
- Distributable app: none of the above; keep the explicit listening-session UX
  (Start/Stop, persistent notification) so the indicator only appears during
  user-initiated sessions.

## Sources

- AOSP, "Privacy indicators": https://source.android.com/docs/core/permissions/privacy-indicators
- AOSP, "Sound Trigger": https://source.android.com/docs/core/audio/sound-trigger
- GreenDotHide (LSPosed module repo): https://modules.lsposed.org/module/com.dorian15.greendothide
- XDA, "How to deactivate the green sensor indicator" (Location Indicator Whitelist): https://xdaforums.com/t/how-do-deactivate-the-green-sensor-indicator-microphone-and-camera.4388359/
- AmbientMusicMod (SoundTrigger/privileged ambient audio precedent): https://github.com/KieronQuinn/AmbientMusicMod
- Qualcomm Hexagon / TFLite delegate: https://blog.tensorflow.org/2019/12/accelerating-tensorflow-lite-on-qualcomm.html
- Lantronix on Snapdragon Voice Activation (CNN second-stage KWS on DSP): https://www.lantronix.com/blog/leverage-qualcomms-socs-voice-control-capabilities-system-modules-lantronix
- GrapheneOS issue #7176 (third-party access to always-on DSP hotword detection): https://github.com/GrapheneOS/os-issue-tracker/issues/7176
