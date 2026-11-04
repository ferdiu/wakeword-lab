package dev.wakeword.lab

import org.junit.Assert.assertEquals
import org.junit.Test

class DetectorConfigTest {
    @Test fun defaultsMatchExportedAudioCadence() {
        val config = DetectorConfig()
        assertEquals(16_000, config.sampleRate)
        assertEquals(16_000, config.windowSamples)
        assertEquals(160, config.captureChunkSamples)
        assertEquals(320, config.scoreEverySamples)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsWrongSampleRate() {
        DetectorConfig(sampleRate = 8_000)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonIntegralScoreChunkCadence() {
        DetectorConfig(scoreEverySamples = 200)
    }
}
