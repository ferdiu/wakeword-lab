package dev.wakeword.lab

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class RollingLogMelFrontendTest {
    @Test fun contiguousTwentyMillisecondShiftMatchesFullRecomputation() {
        val rolling = RollingLogMelFrontend()
        val firstWindow = FloatArray(LogMelFrontend.WINDOW_SAMPLES) { index ->
            (((index * 37) % 997) - 498) / 2000f
        }
        rolling.extract(firstWindow)
        val secondWindow = firstWindow.copyOfRange(320, 16_000) + FloatArray(320) { index ->
            (((index * 71) % 313) - 156) / 1000f
        }
        val incremental = rolling.extract(secondWindow)
        val reference = LogMelFrontend.extract(secondWindow)
        var maxError = 0f
        incremental.indices.forEach { maxError = maxOf(maxError, abs(incremental[it] - reference[it])) }
        assertTrue("rolling frontend maximum error=$maxError", maxError < 0.00001f)
    }

    @Test fun skippedStepFallsBackToFullRecomputation() {
        val rolling = RollingLogMelFrontend()
        val first = FloatArray(LogMelFrontend.WINDOW_SAMPLES) { it / 16_000f }
        rolling.extract(first)
        val skipped = first.copyOfRange(640, 16_000) + FloatArray(640) { (it % 101) / 100f }
        val actual = rolling.extract(skipped)
        val expected = LogMelFrontend.extract(skipped)
        assertTrue(actual.indices.all { abs(actual[it] - expected[it]) < 0.00001f })
    }
}
