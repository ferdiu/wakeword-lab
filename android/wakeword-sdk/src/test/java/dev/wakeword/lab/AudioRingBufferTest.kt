package dev.wakeword.lab

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioRingBufferTest {
    @Test fun wrapsAndKeepsNewestSamplesInChronologicalOrder() {
        val buffer = AudioRingBuffer(4)
        buffer.push(floatArrayOf(.1f, .2f, .3f))
        buffer.push(floatArrayOf(.4f, .5f))
        assertArrayEquals(floatArrayOf(.2f, .3f, .4f, .5f), buffer.snapshot(), 0f)
        assertTrue(buffer.isFull())
    }

    @Test fun leftPadsUntilFull() {
        val buffer = AudioRingBuffer(4)
        buffer.push(floatArrayOf(.5f))
        assertArrayEquals(floatArrayOf(0f, 0f, 0f, .5f), buffer.snapshot(), 0f)
        assertFalse(buffer.isFull())
    }

    @Test fun clipsOutOfRangePcm() {
        val buffer = AudioRingBuffer(2)
        buffer.push(floatArrayOf(-3f, 2f))
        assertArrayEquals(floatArrayOf(-1f, 1f), buffer.snapshot(), 0f)
    }
}
