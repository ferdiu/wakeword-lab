package dev.wakeword.lab

import org.junit.Test

class ThresholdValidationTest {
    @Test fun acceptsValidThresholds() {
        validateThreshold(0f)
        validateThreshold(0.45f)
        validateThreshold(1f)
    }

    @Test fun acceptsNullToClearOverride() {
        validateThreshold(null)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNegativeThreshold() {
        validateThreshold(-0.1f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsThresholdAboveOne() {
        validateThreshold(1.1f)
    }
}
