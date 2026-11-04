package dev.wakeword.lab

/** Thread-safe one-second PCM ring. push() is called by capture; snapshot() by inference. */
class AudioRingBuffer(private val capacity: Int) {
    private val data = FloatArray(capacity)
    private var write = 0
    private var count = 0
    init { require(capacity > 0) }
    @Synchronized fun push(samples: FloatArray) {
        for (v in samples) { data[write] = v.coerceIn(-1f, 1f); write = (write + 1) % capacity; count = (count + 1).coerceAtMost(capacity) }
    }
    @Synchronized fun snapshot(): FloatArray {
        val out = FloatArray(capacity); val missing = capacity-count
        val start = if (count == capacity) write else 0
        for (i in 0 until count) out[missing+i] = data[(start+i)%capacity]
        return out
    }
    @Synchronized fun isFull() = count == capacity
}
