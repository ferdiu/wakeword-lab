package dev.wakeword.lab

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/** Python-compatible 16 kHz mono -> natural-log mel-power frontend, shape [98, 40]. */
object LogMelFrontend {
    const val SAMPLE_RATE = 16_000
    const val WINDOW_SAMPLES = 16_000
    const val FRAME_LENGTH = 400
    const val FRAME_HOP = 160
    const val FFT_LENGTH = 512
    const val MEL_BINS = 40
    const val FRAME_COUNT = 98
    const val FEATURE_COUNT = FRAME_COUNT * MEL_BINS
    private const val F_MIN = 60.0
    private const val F_MAX = 7_600.0
    private val hann = DoubleArray(FRAME_LENGTH) { i ->
        0.5 - 0.5 * cos(2.0 * PI * i / (FRAME_LENGTH - 1))
    }
    private val bank: Array<DoubleArray> = createMelBank()
    private val bitReverse = IntArray(FFT_LENGTH).also { order ->
        var j = 0
        for (i in 1 until FFT_LENGTH) {
            var bit = FFT_LENGTH shr 1
            while ((j and bit) != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            order[i] = j
        }
    }

    /** Returns flattened row-major [98,40] values. Input must be exactly 16,000 samples. */
    fun extract(pcm: FloatArray): FloatArray {
        val result = FloatArray(FEATURE_COUNT)
        extractFrames(pcm, 0, FRAME_COUNT, result, 0)
        return result
    }

    /** Calculate a contiguous subset of spectrogram frames into a row-major destination. */
    internal fun extractFrames(
        pcm: FloatArray,
        firstFrame: Int,
        frameCount: Int,
        destination: FloatArray,
        destinationFirstFrame: Int,
    ) {
        require(pcm.size == WINDOW_SAMPLES) { "Expected $WINDOW_SAMPLES samples, got ${pcm.size}" }
        require(firstFrame >= 0 && frameCount >= 0 && firstFrame + frameCount <= FRAME_COUNT)
        require(destinationFirstFrame >= 0 && destinationFirstFrame + frameCount <= FRAME_COUNT)
        require(destination.size >= (destinationFirstFrame + frameCount) * MEL_BINS)
        val real = DoubleArray(FFT_LENGTH)
        val imag = DoubleArray(FFT_LENGTH)
        val power = DoubleArray(FFT_LENGTH / 2 + 1)

        for (localFrame in 0 until frameCount) {
            val frame = firstFrame + localFrame
            real.fill(0.0); imag.fill(0.0)
            val offset = frame * FRAME_HOP
            for (i in 0 until FRAME_LENGTH) {
                // Match numpy's symmetric np.hanning(400), cast to float32 before multiplication.
                val win = hann[i].toFloat()
                real[i] = (pcm[offset + i] * win).toDouble()
            }
            fftInPlace(real, imag)
            for (k in power.indices) power[k] = real[k] * real[k] + imag[k] * imag[k]

            val outOffset = (destinationFirstFrame + localFrame) * MEL_BINS
            for (m in 0 until MEL_BINS) {
                var energy = 0.0
                val weights = bank[m]
                for (k in weights.indices) energy += power[k] * weights[k]
                destination[outOffset + m] = ln(max(energy, 1e-6)).toFloat()
            }
        }
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * kotlin.math.log10(1.0 + hz / 700.0)

    private fun createMelBank(): Array<DoubleArray> {
        val melMin = hzToMel(F_MIN)
        val melMax = hzToMel(F_MAX)
        val hzPoints = DoubleArray(MEL_BINS + 2) { index ->
            val mel = melMin + (melMax - melMin) * index / (MEL_BINS + 1)
            700.0 * (10.0.pow(mel / 2595.0) - 1.0)
        }
        val bins = IntArray(hzPoints.size) { floor((FFT_LENGTH + 1) * hzPoints[it] / SAMPLE_RATE).toInt() }
        val result = Array(MEL_BINS) { DoubleArray(FFT_LENGTH / 2 + 1) }
        for (m in 1..MEL_BINS) {
            val left = bins[m - 1]; val center = bins[m]; val right = bins[m + 1]
            if (center > left) {
                for (k in max(0, left) until minOf(center, result[m - 1].size)) {
                    result[m - 1][k] = (k - left).toDouble() / (center - left)
                }
            }
            if (right > center) {
                for (k in max(0, center) until minOf(right, result[m - 1].size)) {
                    result[m - 1][k] = (right - k).toDouble() / (right - center)
                }
            }
        }
        return result
    }

    private fun fftInPlace(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        for (i in 0 until n) {
            val j = bitReverse[i]
            if (i < j) {
                val r = real[i]; real[i] = real[j]; real[j] = r
                val im = imag[i]; imag[i] = imag[j]; imag[j] = im
            }
        }
        var length = 2
        while (length <= n) {
            val angle = -2.0 * PI / length
            val stepR = cos(angle); val stepI = sin(angle)
            var base = 0
            while (base < n) {
                var wr = 1.0; var wi = 0.0
                for (j in 0 until length / 2) {
                    val even = base + j; val odd = even + length / 2
                    val vr = real[odd] * wr - imag[odd] * wi
                    val vi = real[odd] * wi + imag[odd] * wr
                    real[odd] = real[even] - vr; imag[odd] = imag[even] - vi
                    real[even] += vr; imag[even] += vi
                    val nextR = wr * stepR - wi * stepI
                    wi = wr * stepI + wi * stepR; wr = nextR
                }
                base += length
            }
            length = length shl 1
        }
    }

    private fun Double.pow(power: Double): Double = kotlin.math.exp(power * ln(this))
}


/** Reuses 96 overlapping log-mel frames when a one-second window advances by exactly 20 ms. */
class RollingLogMelFrontend {
    private var previousPcm: FloatArray? = null
    private val features = FloatArray(LogMelFrontend.FEATURE_COUNT)

    /** Returns a stable copy of row-major [98,40] features. */
    fun extract(pcm: FloatArray): FloatArray {
        require(pcm.size == LogMelFrontend.WINDOW_SAMPLES) {
            "Expected ${LogMelFrontend.WINDOW_SAMPLES} samples, got ${pcm.size}"
        }
        val previous = previousPcm
        val overlap = LogMelFrontend.WINDOW_SAMPLES - 2 * LogMelFrontend.FRAME_HOP
        var contiguous = previous != null
        if (previous != null) {
            for (index in 0 until overlap) {
                if (pcm[index] != previous[index + 2 * LogMelFrontend.FRAME_HOP]) {
                    contiguous = false
                    break
                }
            }
        }

        if (contiguous) {
            val reusedValues = (LogMelFrontend.FRAME_COUNT - 2) * LogMelFrontend.MEL_BINS
            System.arraycopy(features, 2 * LogMelFrontend.MEL_BINS, features, 0, reusedValues)
            LogMelFrontend.extractFrames(
                pcm,
                LogMelFrontend.FRAME_COUNT - 2,
                2,
                features,
                LogMelFrontend.FRAME_COUNT - 2,
            )
        } else {
            LogMelFrontend.extract(pcm).copyInto(features)
        }
        previousPcm = pcm.copyOf()
        return features.copyOf()
    }

    fun clear() {
        previousPcm = null
        features.fill(0f)
    }
}
