package dev.wakeword.lab

data class DetectorConfig(
    val sampleRate: Int = 16000,
    val windowSamples: Int = 16000,
    val chunkSamples: Int = 160,
    val scoreEverySamples: Int = 320,
    val energyThreshold: Float = 0.008f,
    val cooldownMs: Long = 1500
) {
    init {
        require(sampleRate == 16000 && windowSamples == sampleRate)
        require(chunkSamples > 0 && scoreEverySamples >= chunkSamples)
        require(energyThreshold >= 0f && cooldownMs >= 0)
    }
}
