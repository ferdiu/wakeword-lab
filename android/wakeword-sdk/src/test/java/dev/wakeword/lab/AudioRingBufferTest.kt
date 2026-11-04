package dev.wakeword.lab
import org.junit.Assert.*
import org.junit.Test
class AudioRingBufferTest {
 @Test fun wrapsKeepingNewestSamples() { val b=AudioRingBuffer(4); b.push(floatArrayOf(.1f,.2f,.3f)); b.push(floatArrayOf(.4f,.5f)); assertArrayEquals(floatArrayOf(.2f,.3f,.4f,.5f),b.snapshot(),0.00001f) }
 @Test fun padsUntilFull() { val b=AudioRingBuffer(4); b.push(floatArrayOf(.5f)); assertArrayEquals(floatArrayOf(0f,0f,0f,.5f),b.snapshot(),0f) }
}
