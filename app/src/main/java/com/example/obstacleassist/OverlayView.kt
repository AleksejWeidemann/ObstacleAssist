package com.example.obstacleassist

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

// Wird von MainActivity genutzt
data class Detection(
    val box: RectF,
    val score: Float,
    val label: String
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // Thread-sicherer Snapshot
    @Volatile private var detections: List<Detection> = emptyList()

    // Debug-Zeilen (oben/unten)
    @Volatile private var debugTop: String? = null
    @Volatile private var debugBottom: String? = null

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
        color = Color.GREEN
    }

    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        textSize = 42f
        isAntiAlias = true
        color = Color.GREEN
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(160, 0, 0, 0)
        isAntiAlias = true
    }

    fun setDetections(list: List<Detection>) {
        detections = list
        postInvalidateOnAnimation()
    }






    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        // Debug oben
        debugTop?.let { line ->
            drawDebugBox(canvas, line, alignBottom = false)
        }

        // Debug unten
        debugBottom?.let { line ->
            drawDebugBox(canvas, line, alignBottom = true)
        }

        // Draw detections
        val local = detections
        for (d in local) {
            val r = clampRect(d.box, w, h)
            if (r.width() < 4f || r.height() < 4f) continue

            val c = colorForLabel(d.label)
            boxPaint.color = c
            textPaint.color = c

            canvas.drawRect(r, boxPaint)

            val label = "${d.label} ${(d.score * 100).toInt()}%"
            drawLabel(canvas, r, label, c)
        }
    }

    private fun drawDebugBox(canvas: Canvas, text: String, alignBottom: Boolean) {
        val pad = 16f
        val lines = text.split("\n")
        val lineH = textPaint.textSize + 8f

        var maxW = 0f
        for (l in lines) maxW = max(maxW, textPaint.measureText(l))

        val boxW = maxW + 2 * pad
        val boxH = lines.size * lineH + 2 * pad

        val left = pad
        val top = if (!alignBottom) {
            pad
        } else {
            max(pad, height.toFloat() - boxH - pad)
        }

        val rect = RectF(left, top, min(width.toFloat() - pad, left + boxW), top + boxH)
        canvas.drawRoundRect(rect, 12f, 12f, bgPaint)

        var y = rect.top + pad + textPaint.textSize
        for (l in lines) {
            canvas.drawText(l, rect.left + pad, y, textPaint)
            y += lineH
        }
    }

    private fun drawLabel(canvas: Canvas, r: RectF, text: String, color: Int) {
        val pad = 10f
        val textW = textPaint.measureText(text)
        val textH = textPaint.textSize

        val left = r.left
        val top = max(0f, r.top - (textH + 2 * pad))

        val bg = RectF(
            left,
            top,
            min(width.toFloat(), left + textW + 2 * pad),
            top + textH + 2 * pad
        )

        bgPaint.color = Color.argb(160, 0, 0, 0)
        canvas.drawRoundRect(bg, 10f, 10f, bgPaint)

        textPaint.color = color
        canvas.drawText(text, bg.left + pad, bg.bottom - pad, textPaint)
    }

    private fun clampRect(r: RectF, w: Float, h: Float): RectF {
        return RectF(
            r.left.coerceIn(0f, w),
            r.top.coerceIn(0f, h),
            r.right.coerceIn(0f, w),
            r.bottom.coerceIn(0f, h)
        )
    }

    private fun colorForLabel(label: String): Int {
        val hash = label.hashCode()
        val rr = 80 + (hash and 0x7F)
        val gg = 80 + ((hash shr 8) and 0x7F)
        val bb = 80 + ((hash shr 16) and 0x7F)
        return Color.rgb(rr, gg, bb)
    }
}
