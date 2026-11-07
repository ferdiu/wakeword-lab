package dev.wakeword.recorder

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Rect
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.GridLayout
import android.widget.LinearLayout
import android.widget.TextView
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Push-to-talk sample capture: each button records one take into
 *  recordings/<kind>/<split>/ under the app-specific external files dir. */
class MainActivity : Activity() {

    private data class Target(val kind: String, val split: String) {
        val label: String
            get() = "${kind.take(3).uppercase(Locale.US)}·$split"
    }

    private lateinit var status: TextView
    private lateinit var counts: TextView

    private var audioRecord: AudioRecord? = null
    private var captureThread: Thread? = null
    private var pcmBuffer: ByteArrayOutputStream? = null
    private var recordingTarget: Target? = null
    private var startElapsedMs: Long = 0
    private val stopRequested = AtomicBoolean(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val header = TextView(this).apply {
            text = "Training wake word: ${BuildConfig.WAKE_WORD}\n" +
                "Hold a button to record; release to save; slide off to cancel."
            textSize = 16f
            gravity = Gravity.CENTER
            setPadding(24, 24, 24, 16)
        }
        status = TextView(this).apply {
            text = "Ready."
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(24, 8, 24, 8)
        }
        counts = TextView(this).apply {
            textSize = 12f
            gravity = Gravity.CENTER
            setPadding(24, 8, 24, 24)
        }

        // Three columns: the positive row first, then the negative row.
        val grid = GridLayout(this).apply {
            columnCount = 3
            useDefaultMargins = true
        }
        for (target in TARGETS) {
            val button = Button(this).apply { text = target.label }
            button.setOnTouchListener { view, event -> handleTouch(target, view, event) }
            val cell = GridLayout.LayoutParams().apply {
                width = 0
                height = GridLayout.LayoutParams.WRAP_CONTENT
                columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f)
            }
            grid.addView(button, cell)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(header, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(status, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(grid, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(counts, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        setContentView(layout)
        refreshCounts()
    }

    override fun onResume() {
        super.onResume()
        refreshCounts()
    }

    override fun onDestroy() {
        // A recording can only exist while a finger is held down; release the
        // mic if the activity is torn down mid-take rather than leaking it.
        stopRequested.set(true)
        captureThread?.join(1000)
        audioRecord?.let {
            runCatching { it.stop() }
            it.release()
        }
        audioRecord = null
        recordingTarget = null
        super.onDestroy()
    }

    private fun handleTouch(target: Target, view: View, event: MotionEvent): Boolean =
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                beginRecording(target)
                true
            }
            MotionEvent.ACTION_UP -> {
                // Releasing inside the button saves; sliding off cancels.
                endRecording(save = recordingTarget == target && isInside(view, event))
                true
            }
            MotionEvent.ACTION_CANCEL -> {
                endRecording(save = false)
                true
            }
            else -> false
        }

    private fun isInside(view: View, event: MotionEvent): Boolean {
        val rect = Rect()
        view.getGlobalVisibleRect(rect)
        return rect.contains(event.rawX.toInt(), event.rawY.toInt())
    }

    private fun beginRecording(target: Target) {
        if (recordingTarget != null) return // one take at a time; ignore extra presses
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MICROPHONE)
            status.text = "Microphone permission required."
            return
        }
        // VOICE_RECOGNITION matches the detector runtime capture path.
        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuffer, SAMPLE_RATE * 2),
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            status.text = "Microphone unavailable."
            return
        }
        audioRecord = record
        pcmBuffer = ByteArrayOutputStream()
        stopRequested.set(false)
        recordingTarget = target
        startElapsedMs = SystemClock.elapsedRealtime()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        status.text = "● REC ${target.label}"
        captureThread = thread { captureLoop() }
    }

    private fun captureLoop() {
        val record = audioRecord ?: return
        val chunk = ByteArray(2048)
        record.startRecording()
        while (!stopRequested.get()) {
            if (SystemClock.elapsedRealtime() - startElapsedMs >= MAX_DURATION_MS) {
                runOnUiThread { status.text = "Reached the ${MAX_DURATION_MS / 1000}s limit." }
                break
            }
            val read = record.read(chunk, 0, chunk.size)
            if (read > 0) pcmBuffer?.write(chunk, 0, read)
        }
    }

    private fun endRecording(save: Boolean) {
        val target = recordingTarget ?: return
        stopRequested.set(true)
        captureThread?.join(2000)
        captureThread = null
        audioRecord?.let {
            runCatching { it.stop() }
            it.release()
        }
        audioRecord = null
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val durationMs = SystemClock.elapsedRealtime() - startElapsedMs
        recordingTarget = null
        val pcm = pcmBuffer?.toByteArray() ?: ByteArray(0)
        pcmBuffer = null
        when {
            !save -> status.text = "Discarded ${target.label}."
            durationMs < MIN_DURATION_MS -> status.text =
                "Discarded ${target.label} (${durationMs / 1000f}s, need at least ${MIN_DURATION_MS / 1000f}s)."
            else -> writeWav(target, pcm, durationMs)
        }
        refreshCounts()
    }

    private fun writeWav(target: Target, pcm: ByteArray, durationMs: Long) {
        val dir = File(getExternalFilesDir(null), "recordings/${target.kind}/${target.split}")
        dir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val file = File(dir, "rec_${stamp}_${durationMs}.wav")
        file.writeBytes(wavBytes(pcm))
        status.text = "Saved ${target.label} (${durationMs / 1000f}s) → ${file.name}"
    }

    private fun wavBytes(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()

        fun intLE(value: Int) {
            out.write(value and 0xff)
            out.write(value shr 8 and 0xff)
            out.write(value shr 16 and 0xff)
            out.write(value shr 24 and 0xff)
        }
        fun shortLE(value: Int) {
            out.write(value and 0xff)
            out.write(value shr 8 and 0xff)
        }

        out.write("RIFF".toByteArray(Charsets.US_ASCII))
        intLE(36 + pcm.size) // file size minus the RIFF header itself
        out.write("WAVE".toByteArray(Charsets.US_ASCII))
        out.write("fmt ".toByteArray(Charsets.US_ASCII))
        intLE(16) // PCM fmt chunk size
        shortLE(1) // PCM
        shortLE(1) // mono
        intLE(SAMPLE_RATE)
        intLE(SAMPLE_RATE * 2) // byte rate: samples/s * 2 bytes
        shortLE(2) // block align: 1 sample * 2 bytes
        shortLE(16) // bits per sample
        out.write("data".toByteArray(Charsets.US_ASCII))
        intLE(pcm.size)
        out.write(pcm)
        return out.toByteArray()
    }

    private fun refreshCounts() {
        val base = File(getExternalFilesDir(null), "recordings")
        counts.text = TARGETS.joinToString("   ") { target ->
            val n = File(base, "${target.kind}/${target.split}")
                .listFiles { file -> file.extension == "wav" }?.size ?: 0
            "${target.kind}/${target.split}: $n"
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_MICROPHONE && grantResults.any { it != PackageManager.PERMISSION_GRANTED }) {
            status.text = "Microphone permission is required to record."
        }
    }

    companion object {
        private const val REQUEST_MICROPHONE = 7201
        private const val SAMPLE_RATE = 16000
        private const val MIN_DURATION_MS = 300L
        private const val MAX_DURATION_MS = 10_000L

        private val TARGETS = listOf(
            Target("positive", "train"),
            Target("positive", "validation"),
            Target("positive", "test"),
            Target("negative", "train"),
            Target("negative", "validation"),
            Target("negative", "test"),
        )
    }
}
