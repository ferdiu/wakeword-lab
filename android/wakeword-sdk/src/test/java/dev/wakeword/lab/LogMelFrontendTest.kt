package dev.wakeword.lab

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class LogMelFrontendTest {
    @Test fun matchesPythonGoldenFeatures() {
        val waveform = FloatArray(16_000) { i ->
            val t = i.toDouble() / 16_000.0
            (0.10 * sin(2 * PI * 440.0 * t) + 0.035 * sin(2 * PI * 1_000.0 * t + 0.2)).toFloat()
        }
        val expected = javaClass.getResourceAsStream("/frontend_golden.csv")!!
            .bufferedReader().use { reader -> reader.readLines().flatMap { it.split(",") }.map(String::toFloat) }
        val actual = LogMelFrontend.extract(waveform)
        assertEquals(3_920, actual.size)
        var maxError = 0f
        for (i in actual.indices) maxError = maxOf(maxError, abs(actual[i] - expected[i]))
        assertTrue("Python/Kotlin log-mel maximum error=$maxError", maxError < 0.00001f)
    }
}
