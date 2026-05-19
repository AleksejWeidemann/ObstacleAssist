package com.example.obstacleassist

/**
 * WarnEngine – Logging- und Metrik-Schicht für die Masterarbeit-Evaluation.
 *
 * Die Warn-Entscheidungslogik (DangerScore, Zone, Proximity, MotionHints)
 * ist bewusst in MainActivity.DecisionLayer implementiert, um den
 * DangerScore-basierten Ansatz direkt mit Frame-Metriken zu verknüpfen.
 *
 * Diese Klasse ist verantwortlich für:
 *  1. Frame-Metriken pro Frame empfangen und zwischenspeichern
 *  2. Warn-Events mit allen Entscheidungsgrößen in CSV loggen
 */
class WarnEngine(
    private val warnLogger: CsvLogger? = null,
    private val runId: String = "run_unknown",
    private val decisionVersion: String = "DANGER_SCORE_V1"
) {

    private var warnCount: Long = 0L
    private var lastFlushWallMs: Long = 0L

    // ── Frame-Metriken ──────────────────────────────────────────────

    data class FrameMetrics(
        val frameId: Long,
        val wallMs: Long,
        val fps: Float,
        val preMs: Float,
        val infMs: Float,
        val postMs: Float,
        val e2eMs: Float,
        val detCount: Int,
        val droppedFramesEst: Long
    )

    @Volatile private var lastFrameMetrics: FrameMetrics? = null

    /** Wird pro Frame von MainActivity gesetzt (nachdem pre/inf/post bekannt sind). */
    fun updateFrameMetrics(m: FrameMetrics) {
        lastFrameMetrics = m
    }

    // ── Warn-Event Logging ──────────────────────────────────────────

    /**
     * Loggt ein Warn-Event genau an der Stelle, wo tatsächlich gesprochen wird.
     * Alle Entscheidungsgrößen (DangerScore, ClassPriority, ZoneWeight, k)
     * werden von der aufrufenden Stelle (MainActivity.maybeSpeakThreat) übergeben.
     */
    fun logWarnEvent(
        labelRaw: String,
        labelPretty: String,
        zone: String,
        proximity: String = "",
        motionHint: String = "",
        score: Float,
        areaRatio: Float,
        cooldownMs: Long,
        decisionReason: String = decisionVersion,
        dangerScore: Float = 0f,
        classPriority: Int = 0,
        zoneWeight: Float = 0f,
        k: Float = 0f
    ) {
        val logger = warnLogger ?: return
        val fm = lastFrameMetrics

        val frameId = fm?.frameId ?: -1L
        val fps = fm?.fps ?: 0f
        val preMs = fm?.preMs ?: 0f
        val infMs = fm?.infMs ?: 0f
        val postMs = fm?.postMs ?: 0f
        val e2eMs = fm?.e2eMs ?: (preMs + infMs + postMs)
        val detCount = fm?.detCount ?: 0
        val wallMs = fm?.wallMs ?: System.currentTimeMillis()

        logger.logRow(
            runId,
            wallMs,
            frameId,
            labelRaw,
            labelPretty,
            zone,
            proximity,
            motionHint,
            score,
            areaRatio,
            dangerScore,
            classPriority,
            zoneWeight,
            k,
            cooldownMs,
            decisionReason,
            detCount,
            fps,
            preMs,
            infMs,
            postMs,
            e2eMs
        )
        warnCount += 1
        if (warnCount % 5L == 0L || (wallMs - lastFlushWallMs) >= 2000L) {
            logger.flush()
            lastFlushWallMs = wallMs
        }
    }
}