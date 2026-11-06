package dev.wakeword.lab

/** A threshold override must be a valid sigmoid score (or null to clear the override). */
internal fun validateThreshold(value: Float?) {
    require(value == null || value in 0f..1f) { "thresholdOverride must be in [0,1]" }
}

data class DetectorConfig(
    val sampleRate: Int = 16_000,
    val windowSamples: Int = 16_000,
    val captureChunkSamples: Int = 160,
    val scoreEverySamples: Int = 320,
    val energyThreshold: Float = 0.008f,
    val cooldownMs: Long = 1_500L,
    val modelAsset: String = "wakeword_int8.tflite",
    val metadataAsset: String = "model_metadata.json"
) {
    init {
        require(sampleRate == 16_000) { "The exported model requires 16 kHz mono audio" }
        require(windowSamples == sampleRate) { "The model requires a one-second window" }
        require(captureChunkSamples > 0 && scoreEverySamples >= captureChunkSamples)
        require(scoreEverySamples % captureChunkSamples == 0) { "Score hop must be a whole number of capture chunks" }
        require(energyThreshold >= 0f && energyThreshold.isFinite())
        require(cooldownMs >= 0)
        require(modelAsset.isNotBlank() && metadataAsset.isNotBlank())
    }
}
