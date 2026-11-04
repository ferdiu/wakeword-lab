package dev.wakeword.lab

import org.junit.Assert.assertEquals
import org.junit.Test

class Int8QuantizationTest {
    @Test fun tiesRoundToEvenLikeNumpyRint() {
        val values = floatArrayOf(-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f)
        val expected = intArrayOf(-2, -2, 0, 0, 2, 2)
        values.indices.forEach { index ->
            assertEquals(expected[index].toByte(), Int8Quantization.quantize(values[index], 1f, 0))
        }
    }

    @Test fun affineQuantizationClampsAndRoundTrips() {
        assertEquals(127.toByte(), Int8Quantization.quantize(999f, 0.5f, 0))
        assertEquals((-128).toByte(), Int8Quantization.quantize(-999f, 0.5f, 0))
        assertEquals(0.25f, Int8Quantization.dequantize(3, 0.125f, 1), 0f)
    }
}
