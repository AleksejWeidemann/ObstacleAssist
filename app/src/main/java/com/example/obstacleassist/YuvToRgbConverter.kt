package com.example.obstacleassist

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import kotlin.math.max
import kotlin.math.min

/**
 * Schneller YUV_420_888 → ARGB Converter.
 *
 * Optimierung gegenüber Vorgängerversion:
 *   ByteBuffer-Daten werden per Bulk-Copy (.get(byte[])) in lokale Arrays geladen,
 *   statt pro Pixel ByteBuffer.get(index) aufzurufen.
 *   Jeder .get(index)-Aufruf auf einem DirectByteBuffer ist ein JNI-Call.
 *   Bei 640×480 spart das ~920.000 JNI-Aufrufe pro Frame.
 *
 * Erwartete Latenz: ~10-15ms statt ~60-70ms bei 640×480.
 */
class YuvToRgbConverter(@Suppress("UNUSED_PARAMETER") context: Context) {

    private var argbBuffer: IntArray? = null
    private var yArray: ByteArray? = null
    private var uArray: ByteArray? = null
    private var vArray: ByteArray? = null

    fun toBitmap(image: ImageProxy, out: Bitmap) {
        val w = image.width
        val h = image.height
        require(out.width == w && out.height == h) { "Output bitmap must match ImageProxy size" }

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        // ── Bulk-Copy: ByteBuffer → byte[] (je 1 JNI-Call statt W×H Calls) ──
        val ySize = yBuf.remaining()
        val uSize = uBuf.remaining()
        val vSize = vBuf.remaining()

        val yArr = yArray?.takeIf { it.size >= ySize } ?: ByteArray(ySize).also { yArray = it }
        val uArr = uArray?.takeIf { it.size >= uSize } ?: ByteArray(uSize).also { uArray = it }
        val vArr = vArray?.takeIf { it.size >= vSize } ?: ByteArray(vSize).also { vArray = it }

        yBuf.position(0); yBuf.get(yArr, 0, ySize)
        uBuf.position(0); uBuf.get(uArr, 0, uSize)
        vBuf.position(0); vBuf.get(vArr, 0, vSize)

        // ── YUV → RGB (BT.601) mit Array-Zugriff ──
        val needed = w * h
        val outArgb = argbBuffer?.takeIf { it.size == needed } ?: IntArray(needed).also { argbBuffer = it }

        var outIndex = 0
        for (row in 0 until h) {
            val yRow = row * yRowStride
            val uvRow = (row shr 1) * uvRowStride
            for (col in 0 until w) {
                val y = (yArr[yRow + col].toInt() and 0xFF)
                val uvCol = (col shr 1) * uvPixelStride
                val u = (uArr[uvRow + uvCol].toInt() and 0xFF)
                val v = (vArr[uvRow + uvCol].toInt() and 0xFF)

                val c = y - 16
                val d = u - 128
                val e = v - 128

                var r = (298 * c + 409 * e + 128) shr 8
                var g = (298 * c - 100 * d - 208 * e + 128) shr 8
                var b = (298 * c + 516 * d + 128) shr 8

                r = min(255, max(0, r))
                g = min(255, max(0, g))
                b = min(255, max(0, b))

                outArgb[outIndex++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        out.setPixels(outArgb, 0, w, 0, 0, w, h)
    }
}