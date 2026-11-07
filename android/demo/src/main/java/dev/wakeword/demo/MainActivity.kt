package dev.wakeword.demo

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import dev.wakeword.lab.WakeWordListeningService
import kotlin.math.roundToInt
import org.json.JSONObject

class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var thresholdLabel: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var logView: TextView
    private var pendingThreshold: Float = DEFAULT_THRESHOLD

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingThreshold = readPackagedThreshold()

        status = TextView(this).apply {
            text = "Offline wake-word demo\nListening starts only after you tap Start."
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(24, 16, 24, 16)
        }

        thresholdLabel = TextView(this).apply {
            textSize = 14f
            setPadding(24, 8, 8, 8)
        }
        val thresholdSeek = SeekBar(this).apply {
            max = ((THRESHOLD_MAX - THRESHOLD_MIN) * 100).roundToInt()
            progress = ((pendingThreshold - THRESHOLD_MIN) * 100).roundToInt()
            updateThresholdLabel()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    if (!fromUser) return
                    pendingThreshold = THRESHOLD_MIN + progress * 0.01f
                    updateThresholdLabel()
                    // Live-apply when running; keep the companion copy so the next start picks it up.
                    WakeWordListeningService.pendingThresholdOverride = pendingThreshold
                    WakeWordListeningService.active?.setThreshold(pendingThreshold)
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        val thresholdRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(thresholdLabel, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(thresholdSeek, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }

        logView = TextView(this).apply {
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setPadding(16, 8, 16, 8)
        }
        logScroll = ScrollView(this).apply {
            addView(logView, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT))
        }

        val start = Button(this).apply { text = "Start listening"; setOnClickListener { startListening() } }
        val stop = Button(this).apply {
            text = "Stop listening"
            setOnClickListener { stopService(Intent(this@MainActivity, WakeWordListeningService::class.java)); status.text = "Stopped" }
        }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            addView(logScroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(status, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(thresholdRow, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(start, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
            addView(stop, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        setContentView(layout)
    }

    override fun onResume() {
        super.onResume()
        // Detector callbacks already arrive on the main thread; runOnUiThread keeps this safe regardless.
        WakeWordListeningService.uiListener = { line -> runOnUiThread { appendLog(line) } }
    }

    override fun onPause() {
        WakeWordListeningService.uiListener = null
        super.onPause()
    }

    private fun updateThresholdLabel() {
        thresholdLabel.text = "threshold: %.2f".format(pendingThreshold)
    }

    private fun appendLog(line: String) {
        val lines = (logView.text.toString() + line + "\n").lines()
        logView.text = lines.takeLast(MAX_LOG_LINES).joinToString("\n")
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    /** Initial slider position from the packaged model metadata, clamped to the slider range. */
    private fun readPackagedThreshold(): Float = runCatching {
        assets.open("model_metadata.json").bufferedReader().use {
            JSONObject(it.readText()).getDouble("threshold").toFloat()
        }
    }.getOrNull()?.coerceIn(THRESHOLD_MIN, THRESHOLD_MAX) ?: DEFAULT_THRESHOLD

    private fun startListening() {
        val missing = mutableListOf<String>()
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missing += Manifest.permission.RECORD_AUDIO
        }
        // Android 13+ hides foreground-service notifications without this runtime
        // permission; detections are surfaced through that notification.
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            missing += Manifest.permission.POST_NOTIFICATIONS
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), REQUEST_MICROPHONE)
            return
        }
        val intent = Intent(this, WakeWordListeningService::class.java)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent) else startService(intent)
            status.text = "Starting… Keep the foreground notification visible while listening."
        } catch (error: Exception) {
            status.text = "Cannot start microphone service: ${error.message}"
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_MICROPHONE) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) startListening()
            else status.text = "Microphone (and notification) permission is required to listen."
        }
    }

    companion object {
        private const val REQUEST_MICROPHONE = 7101
        private const val THRESHOLD_MIN = 0.20f
        private const val THRESHOLD_MAX = 0.90f
        private const val DEFAULT_THRESHOLD = 0.45f
        private const val MAX_LOG_LINES = 200
    }
}
