package dev.wakeword.lab

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.util.Log
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt
import kotlin.math.max

/**
 * Offline wake-word detector. Audio capture never waits on feature extraction or inference.
 * Call start only from a visible, user-initiated flow after RECORD_AUDIO permission is granted.
 */
class WakeWordDetector(
    context: Context,
    private val listener: Listener,
    private val config: DetectorConfig = DetectorConfig()
) : Closeable {
    private companion object { const val TAG = "WakeWordDetector" }

    interface Listener {
        fun onDetected(score: Float)
        fun onScores(maxScore: Float, threshold: Float) {}
        fun onError(error: Throwable)
    }

    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val modelExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "wakeword-model") }
    private val captureExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "wakeword-audio") }
    private val ring = AudioRingBuffer(config.windowSamples)
    private val lifecycleLock = Any()
    private val running = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val generation = AtomicInteger(0)
    private val inferencePending = AtomicBoolean(false)
    private val lastDetectionElapsedMs = AtomicLong(Long.MIN_VALUE)
    @Volatile private var audioRecord: AudioRecord? = null
    // model is read/written exclusively on modelExecutor.
    private var model: WakeWordModel? = null
    // Live detection-threshold override applied ahead of the model's packaged threshold.
    @Volatile var thresholdOverride: Float? = null
        set(value) {
            validateThreshold(value)
            field = value
        }
    // Debug heartbeat state, touched only on modelExecutor.
    private var inferenceCount = 0
    private var maxScoreSinceLog = 0f

    fun start() {
        synchronized(lifecycleLock) {
            check(!closed.get()) { "Detector is closed" }
            check(appContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "RECORD_AUDIO permission has not been granted"
            }
            if (!running.compareAndSet(false, true)) return
            ring.clear()
            lastDetectionElapsedMs.set(Long.MIN_VALUE)
            val token = generation.incrementAndGet()
            modelExecutor.execute {
                try {
                    if (model == null) model = WakeWordModel(appContext, config)
                    if (running.get() && generation.get() == token) {
                        captureExecutor.execute { captureLoop(token) }
                    }
                } catch (error: Throwable) {
                    // A previous start may fail after stop()+start() created a newer session.
                    if (generation.get() == token) {
                        running.set(false)
                        postError(token, error)
                    }
                }
            }
        }
    }

    fun stop() {
        synchronized(lifecycleLock) {
            if (!running.getAndSet(false)) return
            generation.incrementAndGet()
            runCatching { audioRecord?.stop() }
        }
    }

    fun isListening(): Boolean = running.get()

    @SuppressLint("MissingPermission")
    private fun captureLoop(token: Int) {
        var recorder: AudioRecord? = null
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            val minBuffer = AudioRecord.getMinBufferSize(
                config.sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            check(minBuffer > 0) { "AudioRecord does not support 16 kHz mono PCM16 (code $minBuffer)" }
            val chunkBytes = config.captureChunkSamples * 2
            recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                config.sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(minBuffer * 2, chunkBytes * 8)
            )
            check(recorder.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }
            audioRecord = recorder
            if (!running.get() || generation.get() != token) return
            recorder.startRecording()
            val pcm16 = ShortArray(config.captureChunkSamples)
            var sinceScore = 0
            var energySquares = 0.0
            var energySamples = 0
            while (running.get() && generation.get() == token) {
                val count = recorder.read(pcm16, 0, pcm16.size, AudioRecord.READ_BLOCKING)
                if (count < 0) throw IllegalStateException("AudioRecord.read failed: $count")
                if (count == 0) continue
                val chunk = FloatArray(count)
                for (i in 0 until count) {
                    val value = pcm16[i] / 32768f
                    chunk[i] = value
                    energySquares += value.toDouble() * value.toDouble()
                }
                energySamples += count
                ring.push(chunk)
                sinceScore += count
                if (sinceScore >= config.scoreEverySamples) {
                    sinceScore -= config.scoreEverySamples
                    val rms = if (energySamples == 0) 0f else sqrt((energySquares / energySamples).toFloat())
                    energySquares = 0.0; energySamples = 0
                    if (rms >= config.energyThreshold && ring.isFull() && inferencePending.compareAndSet(false, true)) {
                        val window = ring.snapshot()
                        modelExecutor.execute {
                            try {
                                if (running.get() && generation.get() == token) {
                                    val score = model?.scorePcm(window) ?: return@execute
                                    maxScoreSinceLog = max(maxScoreSinceLog, score)
                                    val effectiveThreshold = thresholdOverride ?: model?.threshold
                                    if (++inferenceCount >= 100) {
                                        inferenceCount = 0
                                        val heartbeatMax = maxScoreSinceLog
                                        maxScoreSinceLog = 0f
                                        Log.d(TAG, "heartbeat: maxScore=${"%.3f".format(heartbeatMax)} threshold=$effectiveThreshold")
                                        mainHandler.post {
                                            if (running.get() && generation.get() == token) {
                                                listener.onScores(heartbeatMax, effectiveThreshold ?: 1f)
                                            }
                                        }
                                    }
                                    val now = SystemClock.elapsedRealtime()
                                    val previous = lastDetectionElapsedMs.get()
                                    if (score >= (effectiveThreshold ?: 1f) &&
                                        (previous == Long.MIN_VALUE || now - previous >= config.cooldownMs)) {
                                        Log.i(TAG, "wake word detected, score=${"%.3f".format(score)}")
                                        lastDetectionElapsedMs.set(now)
                                        mainHandler.post {
                                            if (running.get() && generation.get() == token) listener.onDetected(score)
                                        }
                                    }
                                }
                            } catch (error: Throwable) {
                                // Never let an inference failure from an obsolete session stop a newer one.
                                if (running.get() && generation.get() == token) {
                                    running.set(false)
                                    postError(token, error)
                                }
                            } finally {
                                inferencePending.set(false)
                            }
                        }
                    }
                }
            }
        } catch (error: Throwable) {
            if (running.get() && generation.get() == token) {
                running.set(false)
                postError(token, error)
            }
        } finally {
            if (audioRecord === recorder) audioRecord = null
            runCatching { recorder?.stop() }
            runCatching { recorder?.release() }
        }
    }

    private fun postError(token: Int, error: Throwable) {
        Log.e(TAG, "detector error", error)
        mainHandler.post {
            if (generation.get() == token) listener.onError(error)
        }
    }

    override fun close() {
        synchronized(lifecycleLock) {
            if (!closed.compareAndSet(false, true)) return
            // Invalidate pending callbacks even when capture previously stopped on an error.
            running.set(false)
            generation.incrementAndGet()
            runCatching { audioRecord?.stop() }
            captureExecutor.shutdownNow()
            // Model and all TensorBuffers are closed on their owning executor after queued work.
            modelExecutor.execute {
                runCatching { model?.close() }
                model = null
            }
            modelExecutor.shutdown()
        }
    }
}
