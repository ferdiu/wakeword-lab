package dev.wakeword.lab

import android.content.Context
import android.os.SystemClock
import kotlin.math.roundToInt

/** Run from a background thread. Times model load, frontend, and end-to-end inference. */
object WakeWordModelBenchmark {
    data class Result(
        val modelLoadMs: Double,
        val frontendP50Ms: Double,
        val inferenceP50Ms: Double,
        val endToEndP50Ms: Double,
        val endToEndP90Ms: Double,
        val endToEndP99Ms: Double,
        val modelSizeBytes: Long,
        val iterations: Int
    )

    fun run(context: Context, pcmWindows: List<FloatArray>, warmup: Int = 10): Result {
        require(pcmWindows.isNotEmpty()) { "Provide at least one PCM benchmark window" }
        require(pcmWindows.all { it.size == LogMelFrontend.WINDOW_SAMPLES })
        val startLoad = SystemClock.elapsedRealtimeNanos()
        val model = WakeWordModel(context, DetectorConfig())
        val loadMs = (SystemClock.elapsedRealtimeNanos() - startLoad) / 1e6
        try {
            repeat(warmup.coerceAtLeast(0)) { model.scorePcm(pcmWindows[it % pcmWindows.size]) }
            val frontend = ArrayList<Double>()
            val inference = ArrayList<Double>()
            val total = ArrayList<Double>()
            val rollingFrontend = RollingLogMelFrontend()
            pcmWindows.forEach { pcm ->
                var t = SystemClock.elapsedRealtimeNanos()
                val features = rollingFrontend.extract(pcm)
                val featureTime = SystemClock.elapsedRealtimeNanos() - t
                t = SystemClock.elapsedRealtimeNanos()
                model.scoreFeatures(features)
                val inferenceTime = SystemClock.elapsedRealtimeNanos() - t
                frontend += featureTime / 1e6
                inference += inferenceTime / 1e6
                total += (featureTime + inferenceTime) / 1e6
            }
            val assetSize = context.assets.open("wakeword_int8.tflite").use { input ->
                val buffer = ByteArray(8 * 1024)
                var total = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    total += read
                }
                total
            }
            return Result(
                modelLoadMs = loadMs,
                frontendP50Ms = percentile(frontend, .50),
                inferenceP50Ms = percentile(inference, .50),
                endToEndP50Ms = percentile(total, .50),
                endToEndP90Ms = percentile(total, .90),
                endToEndP99Ms = percentile(total, .99),
                modelSizeBytes = assetSize,
                iterations = pcmWindows.size
            )
        } finally {
            model.close()
        }
    }

    private fun percentile(values: List<Double>, quantile: Double): Double {
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * quantile).roundToInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }
}
