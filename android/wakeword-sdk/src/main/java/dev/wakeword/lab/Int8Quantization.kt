package dev.wakeword.lab

import kotlin.math.round

/** Quantization helpers matching NumPy's float32 np.rint (ties-to-even) semantics. */
internal object Int8Quantization {
    fun quantize(value: Float, scale: Float, zeroPoint: Int): Byte {
        require(scale > 0f && scale.isFinite()) { "Quantization scale must be finite and positive" }
        require(zeroPoint in -128..127) { "INT8 zero point is outside [-128,127]" }
        val quantized = round(value / scale + zeroPoint.toFloat()).toInt().coerceIn(-128, 127)
        return quantized.toByte()
    }

    fun dequantize(value: Byte, scale: Float, zeroPoint: Int): Float {
        require(scale > 0f && scale.isFinite()) { "Quantization scale must be finite and positive" }
        require(zeroPoint in -128..127) { "INT8 zero point is outside [-128,127]" }
        return (value.toInt() - zeroPoint) * scale
    }
}
