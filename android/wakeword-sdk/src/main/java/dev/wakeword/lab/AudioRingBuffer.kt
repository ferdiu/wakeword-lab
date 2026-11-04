package dev.wakeword.lab

/** Fixed-size, synchronized circular PCM buffer. Snapshot is chronological and left-padded. */
class AudioRingBuffer(private val capacity: Int) {
    private val data = FloatArray(capacity)
    private var writeIndex = 0
    private var count = 0

    init { require(capacity > 0) { "capacity must be positive" } }

    @Synchronized
    fun push(samples: FloatArray, length: Int = samples.size) {
        require(length in 0..samples.size)
        for (i in 0 until length) {
            data[writeIndex] = samples[i].coerceIn(-1f, 1f)
            writeIndex = (writeIndex + 1) % capacity
            count = (count + 1).coerceAtMost(capacity)
        }
    }

    @Synchronized
    fun snapshot(): FloatArray {
        val out = FloatArray(capacity)
        val missing = capacity - count
        val start = if (count == capacity) writeIndex else 0
        for (i in 0 until count) out[missing + i] = data[(start + i) % capacity]
        return out
    }

    @Synchronized fun isFull(): Boolean = count == capacity
    @Synchronized fun size(): Int = count
    @Synchronized fun clear() { writeIndex = 0; count = 0; data.fill(0f) }
}
