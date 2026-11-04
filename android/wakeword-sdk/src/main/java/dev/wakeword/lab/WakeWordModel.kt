package dev.wakeword.lab

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject

/** Owns LiteRT and tensor buffers. Create, infer, and close it on the same serial executor. */
internal class WakeWordModel(context: Context, private val config: DetectorConfig) : AutoCloseable {
    private val appContext = context.applicationContext
    // LiteRT documents one process-wide Environment shared by all model instances.
    private val environment = LiteRtRuntime.environment(appContext)
    private var compiledModel: CompiledModel? = null
    private val inputs = mutableListOf<TensorBuffer>()
    private val outputs = mutableListOf<TensorBuffer>()
    private val frontend = RollingLogMelFrontend()
    private var mean = FloatArray(LogMelFrontend.MEL_BINS)
    private var std = FloatArray(LogMelFrontend.MEL_BINS)
    private var inputScale = 0f
    private var inputZeroPoint = 0
    private var outputScale = 0f
    private var outputZeroPoint = 0
    var threshold: Float = 0.8f
        private set

    init {
        try {
            val metadataText = appContext.assets.open(config.metadataAsset).bufferedReader().use { it.readText() }
            val metadata = JSONObject(metadataText)
            require(metadata.getInt("schema_version") == 1) { "Unsupported model metadata schema" }
            require(metadata.getInt("sample_rate") == LogMelFrontend.SAMPLE_RATE)
            require(metadata.getInt("window_samples") == LogMelFrontend.WINDOW_SAMPLES)
            require(metadata.getInt("frame_length_ms") == 25 && metadata.getInt("frame_step_ms") == 10)
            require(metadata.getInt("fft_length") == LogMelFrontend.FFT_LENGTH)
            require(metadata.getInt("mel_bins") == LogMelFrontend.MEL_BINS)
            require(metadata.getDouble("lower_hz") == 60.0 && metadata.getDouble("upper_hz") == 7600.0)
            require(metadata.getString("window") == "symmetric_hann" && metadata.getString("mel_scale") == "htk")
            require(metadata.getString("model_file") == config.modelAsset) { "Model/metadata asset names do not match" }
            require(metadata.getJSONArray("feature_shape").getInt(0) == LogMelFrontend.FRAME_COUNT)
            require(metadata.getJSONArray("feature_shape").getInt(1) == LogMelFrontend.MEL_BINS)
            mean = jsonFloatArray(metadata.getJSONArray("mean"))
            std = jsonFloatArray(metadata.getJSONArray("std"))
            require(mean.size == LogMelFrontend.MEL_BINS && std.size == LogMelFrontend.MEL_BINS)
            require(mean.all { it.isFinite() } && std.all { it > 0f && it.isFinite() })
            val inputQuant = metadata.getJSONObject("input_quantization")
            val outputQuant = metadata.getJSONObject("output_quantization")
            require(inputQuant.getString("dtype") == "int8" && outputQuant.getString("dtype") == "int8")
            val inputShape = metadata.getJSONArray("input_shape")
            require(inputShape.length() == 4 && inputShape.getInt(0) == 1 &&
                inputShape.getInt(1) == LogMelFrontend.FRAME_COUNT &&
                inputShape.getInt(2) == LogMelFrontend.MEL_BINS && inputShape.getInt(3) == 1) {
                "Unsupported model tensor shape: $inputShape"
            }
            inputScale = inputQuant.getDouble("scale").toFloat()
            inputZeroPoint = inputQuant.getInt("zero_point")
            outputScale = outputQuant.getDouble("scale").toFloat()
            outputZeroPoint = outputQuant.getInt("zero_point")
            require(inputScale > 0f && inputScale.isFinite() && outputScale > 0f && outputScale.isFinite())
            require(inputZeroPoint in -128..127 && outputZeroPoint in -128..127)
            val outputShape = metadata.getJSONArray("output_shape")
            require(outputShape.length() > 0) { "Output tensor shape is missing" }
            var outputElements = 1
            for (i in 0 until outputShape.length()) {
                val dimension = outputShape.getInt(i)
                require(dimension > 0) { "Output tensor dimensions must be positive" }
                outputElements *= dimension
            }
            require(outputElements == 1) { "Expected one sigmoid output, got shape $outputShape" }
            threshold = metadata.getDouble("threshold").toFloat()
            require(threshold.isFinite() && threshold in 0f..1f) { "Threshold must be a finite sigmoid probability" }

            compiledModel = CompiledModel.create(
                appContext.assets,
                config.modelAsset,
                CompiledModel.Options(Accelerator.CPU),
                environment
            )
            inputs.addAll(compiledModel!!.createInputBuffers())
            outputs.addAll(compiledModel!!.createOutputBuffers())
            require(inputs.isNotEmpty() && outputs.isNotEmpty())
            // Warm up before the microphone session begins.
            scorePcm(FloatArray(LogMelFrontend.WINDOW_SAMPLES))
        } catch (error: Throwable) {
            closeResources()
            throw error
        }
    }

    fun scorePcm(pcm: FloatArray): Float = scoreFeatures(frontend.extract(pcm))

    fun scoreFeatures(features: FloatArray): Float {
        check(features.size == LogMelFrontend.FEATURE_COUNT) {
            "Expected ${LogMelFrontend.FEATURE_COUNT} features, got ${features.size}"
        }
        val quantized = ByteArray(features.size)
        for (i in features.indices) {
            val channel = i % LogMelFrontend.MEL_BINS
            val normalized = (features[i] - mean[channel]) / std[channel]
            quantized[i] = Int8Quantization.quantize(normalized, inputScale, inputZeroPoint)
        }
        inputs[0].writeInt8(quantized)
        compiledModel!!.run(inputs, outputs)
        val raw = outputs[0].readInt8()
        check(raw.isNotEmpty()) { "Model returned an empty output tensor" }
        return Int8Quantization.dequantize(raw[0], outputScale, outputZeroPoint).coerceIn(0f, 1f)
    }

    override fun close() = closeResources()

    private fun closeResources() {
        inputs.forEach { runCatching { it.close() } }
        outputs.forEach { runCatching { it.close() } }
        inputs.clear(); outputs.clear()
        frontend.clear()
        runCatching { compiledModel?.close() }
        compiledModel = null
    }

    private fun jsonFloatArray(array: org.json.JSONArray): FloatArray =
        FloatArray(array.length()) { i -> array.getDouble(i).toFloat() }
}

/** Shared process-lifetime LiteRT environment; individual model sessions must not close it. */
private object LiteRtRuntime {
    @Volatile private var shared: Environment? = null

    fun environment(context: Context): Environment = synchronized(this) {
        shared ?: Environment.create(context.applicationContext).also { shared = it }
    }
}
