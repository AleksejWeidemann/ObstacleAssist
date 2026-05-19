package com.example.obstacleassist

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tvLeft: TextView
    private lateinit var tvRight: TextView

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // Overlay Stabilisierung
    private val overlayStabilizer = OverlayStabilizer(window = 3, need = 2, iouThr = 0.35f)

    private class OverlayStabilizer(
        private val window: Int = 3,
        private val need: Int = 2,
        private val iouThr: Float = 0.35f
    ) {
        private val hist: ArrayDeque<List<Detection>> = ArrayDeque(window)

        fun updateAndStabilize(current: List<Detection>): List<Detection> {
            hist.addLast(current)
            while (hist.size > window) hist.removeFirst()
            if (hist.size < need) return emptyList()

            return current.filter { d ->
                var hits = 0
                for (frame in hist) {
                    if (frame.any { it.label == d.label && iou(it.box, d.box) >= iouThr }) hits++
                    if (hits >= need) return@filter true
                }
                false
            }
        }

        private fun iou(a: RectF, b: RectF): Float {
            val x1 = maxOf(a.left, b.left)
            val y1 = maxOf(a.top, b.top)
            val x2 = minOf(a.right, b.right)
            val y2 = minOf(a.bottom, b.bottom)
            val interW = maxOf(0f, x2 - x1)
            val interH = maxOf(0f, y2 - y1)
            val inter = interW * interH
            val areaA = maxOf(0f, a.width()) * maxOf(0f, a.height())
            val areaB = maxOf(0f, b.width()) * maxOf(0f, b.height())
            val union = areaA + areaB - inter
            return if (union <= 0f) 0f else inter / union
        }
    }

    // ORT
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private val modelAssetName = "obstacle_mvp_15c.onnx"
    private val inputSize = 640

    // Wiederverwendbare Puffer (vermeidet GC-Pressure + Allokation pro Frame)
    private var reusableRotatedBmp: Bitmap? = null
    private var reusableLetterboxBmp: Bitmap? = null
    private var reusableScaledBmp: Bitmap? = null
    private var reusableFloatBuffer: FloatBuffer? = null
    private var reusablePixelArray: IntArray? = null
    private var reusableNchwArray: FloatArray? = null

    private var confThreshold = 0.30f
    private val nmsIoU = 0.45f

    // ── Ablation: true = DangerScore, false = nur Confidence (für A/B-Vergleich) ──
    private val useDangerScore = true

    // CSV Logging
    private val runId: String = "run_" + System.currentTimeMillis()
    private lateinit var frameLogger: CsvLogger
    private lateinit var warnLogger: CsvLogger
    private lateinit var warnEngine: WarnEngine

    private var frameId: Long = 0L
    private var lastFrameTsMs: Long = 0L
    private var camW: Int = 0
    private var camH: Int = 0

    // Fix 3: Kamera-Timestamp für echte Drop-Rate
    private var lastCamTimestampNs: Long = 0L
    private var camIntervalSumNs: Long = 0L
    private var camIntervalCount: Long = 0L
    private var droppedFramesEst: Long = 0L   // kumulativ geschätzte verworfene Frames

    // Fix 4: Stall-Erkennung (rolling median über letzte 30 Frames)
    private val recentE2e = ArrayDeque<Float>(32)
    private val stallMultiplier = 2.5f

    // Labels
    private lateinit var classLabels: Array<String>

    private fun fallbackLabels(): Array<String> = arrayOf(
        "Ampel", "Auto", "Baustellenhindernis", "Fahrrad", "Grosses_Fahrzeug",
        "Hund", "Laterne_Pfosten", "Motorrad_E-Scooter", "Muelltonne", "Person",
        "Sitzbank", "Stuhl", "Tisch", "Zaun_Absperrung", "Zebrastreifen"
    )

    private fun loadLabelsOrFallback(): Array<String> {
        return try {
            val lines = assets.open("labels.txt").bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotBlank() }
            if (lines.isNotEmpty()) lines.toTypedArray() else fallbackLabels()
        } catch (t: Throwable) {
            fallbackLabels()
        }
    }

    private fun labelFor(classId: Int): String {
        return if (::classLabels.isInitialized && classId in classLabels.indices) {
            classLabels[classId]
        } else classId.toString()
    }

    private fun prettyLabel(raw: String) = raw.replace("_", " ").replace("-", " / ").trim()
    private fun normLabel(s: String) = s.lowercase(Locale.ROOT).replace(Regex("[_\\s/-]"), "")

    // TTS
    private var tts: TextToSpeech? = null
    private var lastTtsAtMs = 0L

    // Fix A: Klassen-Cooldown – verhindert Dauerbeschallung bei vielen Objekten gleicher Klasse
    private val lastClassWarnMs = HashMap<String, Long>()

    // Fix B: Crowd-Modus – aggregierte Warnung bei Menschenmengen
    private var lastCrowdWarnMs = 0L
    private val crowdCooldownMs = 10000L  // 10s zwischen Crowd-Warnungen
    private val crowdThreshold = 3        // ab 3 Personen in MID-Zone

    // Metrics
    private var lastMetricAt = 0L
    private var frameCount = 0
    private var lastFps: Float = 0f

    // Decision Layer
    private enum class Zone { LEFT, MID, RIGHT }
    private enum class Proximity { FAR, NEAR, IMMEDIATE }
    private enum class MotionHint {
        NONE, COMING_CLOSER, COMING_FROM_LEFT, COMING_FROM_RIGHT,
        IMMEDIATE_FROM_LEFT, IMMEDIATE_FROM_RIGHT
    }

    private data class Threat(
        val labelRaw: String, val labelPretty: String, val score: Float,
        val zone: Zone, val areaRatio: Float, val cxNorm: Float,
        val box: RectF, val prox: Proximity, val motionHint: MotionHint
    )

    private val warnWhitelistNorm = setOf(
        normLabel("Person"), normLabel("Fahrrad"), normLabel("Motorrad_E-Scooter"),
        normLabel("Auto"), normLabel("Grosses_Fahrzeug"), normLabel("Laterne_Pfosten"),
        normLabel("Zaun_Absperrung"), normLabel("Baustellenhindernis"), normLabel("Muelltonne")
    )

    // ── Persistenz-Heuristik: Lead-Objekt-Tracking + Follow-Mode ──────────
    private fun boxIou(a: RectF, b: RectF): Float {
        val interL = max(a.left, b.left); val interT = max(a.top, b.top)
        val interR = min(a.right, b.right); val interB = min(a.bottom, b.bottom)
        val inter = max(0f, interR - interL) * max(0f, interB - interT)
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }

    private class PersistentGate(
        private val iouFn: (RectF, RectF) -> Float,
        private val normLabel: (String) -> String
    ) {
        data class Peek(val labelNorm: String, val zone: Zone, val box: RectF)

        private data class Track(
            var labelNorm: String, var zone: Zone, var box: RectF,
            var firstSeenMs: Long, var lastSeenMs: Long,
            var lastWarnMs: Long, var lastWarnProx: Proximity,
            var lastWarnDanger: Float, var lastWarnArea: Float,
            var warnCount: Int
        )

        data class Decision(val allow: Boolean, val reason: String, val cooldownMs: Long = 0L)

        private var track: Track? = null
        private val matchIou = 0.40f
        private val maxGapMs = 900L
        private val emaAlpha = 0.35f

        private fun smooth(dst: RectF, src: RectF) {
            dst.left = dst.left * (1f - emaAlpha) + src.left * emaAlpha
            dst.top = dst.top * (1f - emaAlpha) + src.top * emaAlpha
            dst.right = dst.right * (1f - emaAlpha) + src.right * emaAlpha
            dst.bottom = dst.bottom * (1f - emaAlpha) + src.bottom * emaAlpha
        }

        fun peek(): Peek? = track?.let { Peek(it.labelNorm, it.zone, RectF(it.box)) }

        fun observe(t: Threat, nowMs: Long) {
            val ln = normLabel(t.labelRaw)
            val tr = track
            if (tr == null) {
                track = Track(ln, t.zone, RectF(t.box), nowMs, nowMs, 0L, Proximity.FAR, 0f, 0f, 0)
                return
            }
            // Zone bewusst NICHT im Matching: Person am Zonenrand (MID↔LEFT)
            // soll denselben Track behalten wenn IoU hoch genug
            val same = (nowMs - tr.lastSeenMs) <= maxGapMs &&
                    tr.labelNorm == ln &&
                    iouFn(tr.box, t.box) >= matchIou
            if (!same) {
                track = Track(ln, t.zone, RectF(t.box), nowMs, nowMs, 0L, Proximity.FAR, 0f, 0f, 0)
            } else {
                tr.lastSeenMs = nowMs
                tr.zone = t.zone   // Zone updaten, Track beibehalten
                smooth(tr.box, t.box)
            }
        }

        fun decide(t: Threat, nowMs: Long, danger: Float): Decision {
            val ln = normLabel(t.labelRaw)
            val tr = track ?: return Decision(true, "PERSIST_INIT", 0L)

            val same = (nowMs - tr.lastSeenMs) <= maxGapMs &&
                    tr.labelNorm == ln &&
                    iouFn(tr.box, t.box) >= matchIou

            if (!same) return Decision(true, "PERSIST_NEW", 0L)

            val sinceWarn = if (tr.lastWarnMs == 0L) Long.MAX_VALUE else (nowMs - tr.lastWarnMs)

            fun rank(p: Proximity) = when (p) {
                Proximity.FAR -> 0; Proximity.NEAR -> 1; Proximity.IMMEDIATE -> 2
            }

            val personNorm = normLabel("Person")
            val isPersonMid = (ln == personNorm && t.zone == Zone.MID)

            // ── FIX 1: Proximity-Hysterese ──
            val proxDelta = rank(t.prox) - rank(tr.lastWarnProx)
            val areaJump = (t.areaRatio - tr.lastWarnArea) >= max(0.015f, tr.lastWarnArea * 0.40f)
            val realEscalation = proxDelta >= 2 || (proxDelta == 1 && areaJump)

            // FAR: nur warnen wenn DangerScore relevant
            if (t.prox == Proximity.FAR && danger < 0.55f) {
                return Decision(false, "PERSIST_FAR_LOW", 0L)
            }

            // ── FIX 2: IMMEDIATE mit längerem Cooldown ──
            if (t.prox == Proximity.IMMEDIATE) {
                val immCooldown = if (isPersonMid) 5000L else 3000L
                if (sinceWarn < immCooldown) return Decision(false, "PERSIST_IMM_SPAM", immCooldown)
                return allow(tr, t, nowMs, danger, "PERSIST_IMMEDIATE", immCooldown)
            }

            // Eskalation: nur bei ECHTER Proximity-Änderung (nicht Flicker)
            if (realEscalation) {
                val escCooldown = 2000L
                if (sinceWarn < escCooldown) return Decision(false, "PERSIST_ESC_CD", escCooldown)
                return allow(tr, t, nowMs, danger, "PERSIST_ESCALATE", escCooldown)
            }

            // Approach: kommt klar näher (area wächst deutlich)
            val approaching = t.motionHint != MotionHint.NONE
            val dangerJump = (danger - tr.lastWarnDanger) >= 0.25f
            if (approaching && dangerJump && areaJump) {
                val approachCooldown = 2000L
                if (sinceWarn < approachCooldown) return Decision(false, "PERSIST_ESC_CD", approachCooldown)
                return allow(tr, t, nowMs, danger, "PERSIST_APPROACH", approachCooldown)
            }

            // ── FIX 3: Follow-Mode mit längeren Reminder-Intervallen ──
            val reminderMs = when {
                isPersonMid -> when (tr.warnCount) {
                    0 -> 0L; 1 -> 6000L; 2 -> 12000L; else -> 20000L
                }
                t.prox == Proximity.NEAR -> when (tr.warnCount) {
                    0 -> 0L; 1 -> 5000L; else -> 10000L
                }
                t.prox == Proximity.FAR -> 15000L
                else -> 10000L
            }

            if (sinceWarn < reminderMs) return Decision(false, "PERSIST_FOLLOW", reminderMs)

            return allow(tr, t, nowMs, danger, "PERSIST_REMINDER", reminderMs)
        }

        private fun allow(tr: Track, t: Threat, nowMs: Long, danger: Float, reason: String, cooldownMs: Long): Decision {
            tr.lastWarnMs = nowMs
            tr.lastWarnProx = t.prox
            tr.lastWarnDanger = danger
            tr.lastWarnArea = t.areaRatio
            tr.warnCount += 1
            return Decision(true, reason, cooldownMs)
        }
    }

    private val persistGate = PersistentGate(::boxIou, ::normLabel)

    private val areaThrMidByClassNorm = mapOf(
        normLabel("Person") to 0.030f, normLabel("Fahrrad") to 0.022f,
        normLabel("Motorrad_E-Scooter") to 0.024f, normLabel("Auto") to 0.028f,
        normLabel("Grosses_Fahrzeug") to 0.030f, normLabel("Baustellenhindernis") to 0.040f,
        normLabel("Zaun_Absperrung") to 0.055f, normLabel("Muelltonne") to 0.045f,
        normLabel("Laterne_Pfosten") to 0.050f
    )

    private val areaThrSideByClassNorm = mapOf(
        normLabel("Person") to 0.050f, normLabel("Fahrrad") to 0.040f,
        normLabel("Motorrad_E-Scooter") to 0.042f, normLabel("Auto") to 0.050f,
        normLabel("Grosses_Fahrzeug") to 0.055f, normLabel("Baustellenhindernis") to 0.070f,
        normLabel("Zaun_Absperrung") to 0.090f, normLabel("Muelltonne") to 0.075f,
        normLabel("Laterne_Pfosten") to 0.080f
    )

    private val defaultAreaThrMid = 0.030f
    private val defaultAreaThrSide = 0.050f

    private fun areaThreshold(labelRaw: String, zone: Zone): Float {
        val n = normLabel(labelRaw)
        return when (zone) {
            Zone.MID -> areaThrMidByClassNorm[n] ?: defaultAreaThrMid
            Zone.LEFT, Zone.RIGHT -> areaThrSideByClassNorm[n] ?: defaultAreaThrSide
        }
    }

    /**
     * Klassenprioritäten: Höher = gefährlicher/wichtiger
     * Unterscheidet Hindernisse (6-10) von Info-Klassen (2-4)
     */
    private fun getClassPriority(label: String): Int {
        return when (label) {
            "Auto" -> 10
            "Grosses_Fahrzeug" -> 10
            "Motorrad_E-Scooter" -> 9
            "Fahrrad" -> 9
            "Person" -> 8
            "Baustellenhindernis" -> 8
            "Muelltonne" -> 7
            "Laterne_Pfosten" -> 7
            "Zaun_Absperrung" -> 6
            "Sitzbank" -> 5
            "Stuhl" -> 5
            "Tisch" -> 5
            "Ampel" -> 6
            "Zebrastreifen" -> 6
            else -> 5
        }
    }

    /**
     * Zone-Gewichtung: MID (vor dir) wichtiger als LEFT/RIGHT
     */
    private fun getZoneWeight(zone: String): Float {
        return when (zone) {
            "MID" -> 1.30f
            "LEFT" -> 1.0f
            "RIGHT" -> 1.0f
            else -> 1.0f
        }
    }

    /**
     * DangerScore-Berechnung
     * Formel: score × (1 + k × area) × zoneWeight × (1 + 0.05 × classPrio)
     * ClassPriority als Tie-Breaker, nicht Haupttreiber (0.05 statt 0.1)
     */
    private fun computeDangerScore(
        score: Float,
        areaRatio: Float,
        zone: String,
        label: String,
        k: Float = 2.5f
    ): Triple<Float, Int, Float> {
        val classPrio = getClassPriority(label)
        val zoneWeight = getZoneWeight(zone)
        val dangerScore = score *
                (1.0f + k * areaRatio) *
                zoneWeight *
                (1.0f + 0.05f * classPrio)
        return Triple(dangerScore, classPrio, zoneWeight)
    }

    private fun zoneFor(cxPx: Float, srcW: Float): Zone {
        val x = cxPx / max(1f, srcW)
        return when {
            x < 0.40f -> Zone.LEFT
            x > 0.60f -> Zone.RIGHT
            else -> Zone.MID
        }
    }

    private fun proximityFor(areaRatio: Float, baseThr: Float) = when {
        areaRatio >= baseThr * 2.2f -> Proximity.IMMEDIATE
        areaRatio >= baseThr * 1.4f -> Proximity.NEAR
        else -> Proximity.FAR
    }

    // Stability-Check entfernt — PersistentGate übernimmt die Warn-Entscheidung

    private class TrendBuffer(private val maxLen: Int = 7) {
        private data class Sample(val tMs: Long, val area: Float, val cx: Float)
        private val map = HashMap<String, ArrayDeque<Sample>>()

        fun add(labelNorm: String, nowMs: Long, areaRatio: Float, cxNorm: Float) {
            val q = map.getOrPut(labelNorm) { ArrayDeque() }
            q.addLast(Sample(nowMs, areaRatio, cxNorm))
            while (q.size > maxLen) q.removeFirst()
        }

        fun hint(labelNorm: String, prox: Proximity): MotionHint {
            val q = map[labelNorm] ?: return MotionHint.NONE
            if (q.size < 6) return MotionHint.NONE

            val first = q.first()
            val last = q.last()
            val areaDelta = last.area - first.area
            val cxDelta = last.cx - first.cx

            val startRight = first.cx > 0.62f
            val startLeft = first.cx < 0.38f
            val comingCloser = areaDelta > max(0.012f, first.area * 0.40f)

            if (!comingCloser) return MotionHint.NONE

            val strongDriftRight = startRight && cxDelta < -0.14f
            val strongDriftLeft = startLeft && cxDelta > 0.14f

            return when {
                prox == Proximity.IMMEDIATE && strongDriftRight -> MotionHint.IMMEDIATE_FROM_RIGHT
                prox == Proximity.IMMEDIATE && strongDriftLeft -> MotionHint.IMMEDIATE_FROM_LEFT
                startRight && cxDelta < -0.08f -> MotionHint.COMING_FROM_RIGHT
                startLeft && cxDelta > 0.08f -> MotionHint.COMING_FROM_LEFT
                else -> MotionHint.COMING_CLOSER
            }
        }
    }

    private val trendBuffer = TrendBuffer(maxLen = 7)

    private object WarnTextBuilder {
        fun buildGerman(labelPretty: String, zone: Zone, prox: Proximity, motion: MotionHint): String {
            // Fix 8d: Kürzere Texte für schnellere Ansage
            val dir = when (zone) {
                Zone.MID -> "voraus"
                Zone.LEFT -> "links"
                Zone.RIGHT -> "rechts"
            }

            return when (prox) {
                Proximity.IMMEDIATE -> "Stopp! $labelPretty $dir!"
                Proximity.NEAR -> {
                    val motionSuffix = when (motion) {
                        MotionHint.COMING_FROM_LEFT -> " Von links."
                        MotionHint.COMING_FROM_RIGHT -> " Von rechts."
                        MotionHint.COMING_CLOSER -> " Kommt näher."
                        else -> ""
                    }
                    "$labelPretty nah $dir.$motionSuffix"
                }
                Proximity.FAR -> "Achtung, $labelPretty $dir."
            }
        }
    }

    private object DecisionLayer {
        data class Result(val best: Threat?, val key: String?, val threatsCount: Int)

        fun selectThreatsAndBest(
            dets: List<Detection>, srcW: Float, srcH: Float, confThreshold: Float,
            warnWhitelistNorm: Set<String>, normLabel: (String) -> String,
            prettyLabel: (String) -> String, zoneFor: (Float, Float) -> Zone,
            areaThreshold: (String, Zone) -> Float, proximityFor: (Float, Float) -> Proximity,
            trendBuffer: TrendBuffer, nowMs: Long,
            computeDangerScore: (Float, Float, String, String) -> Triple<Float, Int, Float>,
            useDangerScore: Boolean = true,
            prevTrack: PersistentGate.Peek? = null,
            iouFn: (RectF, RectF) -> Float = { _, _ -> 0f }
        ): Result {
            val frameArea = max(1f, srcW * srcH)

            val threats = dets.mapNotNull { d ->
                val labelRaw = d.label
                val labelNorm = normLabel(labelRaw)
                if (!warnWhitelistNorm.contains(labelNorm)) return@mapNotNull null
                if (d.score < confThreshold) return@mapNotNull null

                val b = d.box
                val areaRatio = (max(0f, b.width()) * max(0f, b.height())) / frameArea
                val cxNorm = (b.centerX() / max(1f, srcW)).coerceIn(0f, 1f)
                val zone = zoneFor(b.centerX(), srcW)
                val baseThr = areaThreshold(labelRaw, zone)
                if (areaRatio < baseThr) return@mapNotNull null

                val prox = proximityFor(areaRatio, baseThr)
                trendBuffer.add(labelNorm, nowMs, areaRatio, cxNorm)
                val motion = trendBuffer.hint(labelNorm, prox)

                Threat(labelRaw, prettyLabel(labelRaw), d.score, zone, areaRatio,
                    cxNorm, b, prox, motion)
            }

            // Ablation: DangerScore (mit Stickiness-Bonus) vs. reine Confidence
            val best = if (useDangerScore) {
                threats.maxByOrNull { threat ->
                    val (danger, _, _) = computeDangerScore(
                        threat.score, threat.areaRatio, threat.zone.name, threat.labelRaw
                    )
                    // Stickiness: Lead-Objekt bekommt 12% Bonus → verhindert Flicker
                    val sticky = if (prevTrack != null &&
                        prevTrack.labelNorm == normLabel(threat.labelRaw) &&
                        iouFn(prevTrack.box, threat.box) >= 0.40f
                    ) 1.12f else 1.0f
                    danger * sticky
                }
            } else {
                threats.maxByOrNull { it.score }
            }
            val key = best?.let { "${normLabel(it.labelRaw)}|${it.zone}|${it.prox}" }
            return Result(best, key, threats.size)
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        overlayView = OverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        tvLeft = TextView(this).apply {
            textSize = 14f
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            setPadding(24, 24, 24, 24)
            text = "THREAT: -"
        }

        tvRight = TextView(this).apply {
            textSize = 13f
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            setPadding(18, 12, 18, 12)
            maxLines = 2
            text = "STATUS: -"
        }

        val root = FrameLayout(this).apply {
            addView(previewView)
            addView(overlayView)
            addView(tvLeft, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                leftMargin = 16; topMargin = 16
            })
            addView(tvRight, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.END
                rightMargin = 16; bottomMargin = 16
            })
        }

        setContentView(root)

        frameLogger = CsvLogger(this, "frame_metrics_${runId}.csv",
            "run_id;timestamp_ms;frame_id;fps;pre_ms;inf_ms;post_ms;e2e_ms;det_count;dropped_frames_est;cam_w;cam_h;cam_fps;drop_rate_pct;is_stall;device_model;android_version;model_name;input_size;conf_threshold;nms_iou;selection_mode"
        ).apply { open() }

        warnLogger = CsvLogger(this, "warn_events_${runId}.csv",
            "run_id;timestamp_ms;frame_id;label_raw;label_pretty;zone;proximity;motion_hint;score;area_ratio;danger_score;class_priority;zone_weight;k;cooldown_ms;decision_reason;det_count;fps;pre_ms;inf_ms;post_ms;e2e_ms"
        ).apply { open() }

        val selectionMode = if (useDangerScore) "DANGER_SCORE_V1" else "CONFIDENCE_ONLY"
        warnEngine = WarnEngine(warnLogger, runId, selectionMode)

        classLabels = loadLabelsOrFallback()
        tts = TextToSpeech(this, this)

        val modelBytes = assets.open(modelAssetName).readBytes()
        initOrtSimple(modelBytes)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            frameLogger.flush(); warnLogger.flush()
            frameLogger.close(); warnLogger.close()
        } catch (_: Throwable) {}
        try { ortSession?.close() } catch (_: Throwable) {}
        try { ortEnv?.close() } catch (_: Throwable) {}
        try { tts?.stop(); tts?.shutdown() } catch (_: Throwable) {}
        cameraExecutor.shutdown()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.GERMAN
            tts?.setSpeechRate(1.25f)  // 25% schneller für zeitkritische Warnungen
        }
    }

    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .apply {
                    // Elektronische Bildstabilisierung (EIS) via Camera2 Interop
                    try {
                        val ext = androidx.camera.camera2.interop.Camera2Interop.Extender(this)
                        ext.setCaptureRequestOption(
                            android.hardware.camera2.CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                            android.hardware.camera2.CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_ON
                        )
                    } catch (_: Throwable) { Log.w("APP", "EIS not supported") }
                }
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(Size(640, 480))  // Standard-Auflösung, vermeidet 1088×1088 Fallback
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    processFrame(imageProxy)
                } catch (e: Throwable) {
                    Log.e("APP", "Frame error: ${e.message}")
                } finally {
                    imageProxy.close()

                    frameCount++
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastMetricAt >= 1000L) {
                        lastFps = frameCount * 1000f / max(1, (now - lastMetricAt)).toFloat()
                        frameCount = 0
                        lastMetricAt = now
                    }
                }
            }

            val selector = androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Exception) {
                Log.e("APP", "Camera error: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Direkte YUV→RGB Konvertierung (kein JPEG-Encode/Decode)
    private val yuvConverter by lazy { YuvToRgbConverter(this) }
    private var reusableYuvBitmap: Bitmap? = null

    private fun yuvToRgbBitmap(imageProxy: ImageProxy): Bitmap {
        val w = imageProxy.width
        val h = imageProxy.height
        val bmp = reusableYuvBitmap?.takeIf { it.width == w && it.height == h && !it.isRecycled }
            ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { reusableYuvBitmap = it }
        yuvConverter.toBitmap(imageProxy, bmp)
        return bmp
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val nowMs = SystemClock.elapsedRealtime()
        val preStart = System.nanoTime()

        // Fix 3: Kamera-Timestamp für echte Drop-Rate
        val camTs = imageProxy.imageInfo.timestamp  // Nanosekunden
        if (lastCamTimestampNs > 0L && camTs > lastCamTimestampNs) {
            camIntervalSumNs += (camTs - lastCamTimestampNs)
            camIntervalCount += 1
        }
        lastCamTimestampNs = camTs

        // Kamera-Auflösung loggen (einmalig)
        if (frameId == 0L) {
            camW = imageProxy.width
            camH = imageProxy.height
            Log.i("APP", "Camera: ${camW}x${camH} rot=${imageProxy.imageInfo.rotationDegrees} mode=${if (useDangerScore) "DangerScore" else "ConfidenceOnly"}")
        }

        val rgbBitmap = yuvToRgbBitmap(imageProxy)
        val (inputTensor, letterbox) = makeInputTensor(rgbBitmap, imageProxy.imageInfo.rotationDegrees)
        val preEnd = System.nanoTime()

        val infStart = System.nanoTime()
        val rawOutput = runOrt(inputTensor)
        val infEnd = System.nanoTime()

        val postStart = System.nanoTime()
        val candidates = decodeYolo(rawOutput, letterbox)
        val filtered = candidates.filter { it.score >= confThreshold }
        val finalDetsRaw = nmsPerClass(filtered, nmsIoU)
        val finalDets = filterEdgeArtifacts(finalDetsRaw, letterbox.srcW, letterbox.srcH)
        val postEnd = System.nanoTime()

        val prev = persistGate.peek()

        val decision = DecisionLayer.selectThreatsAndBest(
            finalDets, letterbox.srcW.toFloat(), letterbox.srcH.toFloat(),
            confThreshold, warnWhitelistNorm, ::normLabel, ::prettyLabel,
            ::zoneFor, ::areaThreshold, ::proximityFor, trendBuffer, nowMs,
            { score, area, zone, label -> computeDangerScore(score, area, zone, label) },
            useDangerScore, prev, ::boxIou
        )

        val bestThreat = decision.best
        val key = decision.key

        // PersistentGate: Lead-Objekt tracken (auch ohne Warnung)
        if (bestThreat != null) {
            persistGate.observe(bestThreat, nowMs)
        }

        val preMs = ((preEnd - preStart) / 1e6).toFloat()
        val infMs = ((infEnd - infStart) / 1e6).toFloat()
        val postMs = ((postEnd - postStart) / 1e6).toFloat()

        frameId += 1
        val wallMs = System.currentTimeMillis()
        val dtMs = if (lastFrameTsMs == 0L) 0L else (nowMs - lastFrameTsMs)
        val fpsInstant = if (dtMs > 0) 1000f / dtMs.toFloat() else 0f
        lastFrameTsMs = nowMs

        val e2eMs = preMs + infMs + postMs
        val detCount = finalDets.size

        // Fix 3: Kamera-FPS und Drop-Rate berechnen
        val camFps = if (camIntervalCount >= 5) {
            1e9f / (camIntervalSumNs.toFloat() / camIntervalCount.toFloat())
        } else 0f
        val dropRatePct = if (camFps > 0f && fpsInstant > 0f) {
            ((camFps - fpsInstant) / camFps * 100f).coerceIn(0f, 100f)
        } else 0f

        // Geschätzte verworfene Frames seit letztem verarbeiteten Frame
        if (camFps > 0f && dtMs > 0L) {
            val expectedFrames = (dtMs * camFps / 1000f).toLong()
            val dropped = (expectedFrames - 1).coerceAtLeast(0L)
            droppedFramesEst += dropped
        }

        // Fix 4: Stall-Erkennung
        recentE2e.addLast(e2eMs)
        while (recentE2e.size > 30) recentE2e.removeFirst()
        val medianE2e = if (recentE2e.size >= 5) {
            recentE2e.toList().sorted()[recentE2e.size / 2]
        } else e2eMs
        val isStall = if (e2eMs > medianE2e * stallMultiplier && e2eMs > 300f) 1 else 0

        frameLogger.logRow(runId, wallMs, frameId, fpsInstant, preMs, infMs, postMs, e2eMs, detCount, droppedFramesEst,
            camW, camH, camFps, dropRatePct, isStall,
            android.os.Build.MODEL, android.os.Build.VERSION.SDK_INT, modelAssetName, inputSize, confThreshold, nmsIoU,
            if (useDangerScore) "DANGER_SCORE" else "CONFIDENCE_ONLY")
        if (frameId % 30L == 0L) frameLogger.flush()

        warnEngine.updateFrameMetrics(
            WarnEngine.FrameMetrics(frameId, wallMs, fpsInstant, preMs, infMs, postMs, e2eMs, detCount, droppedFramesEst)
        )

        runOnUiThread {
            val vw = overlayView.width.toFloat()
            val vh = overlayView.height.toFloat()
            val cleaned = if (vw > 0f && vh > 0f && letterbox.srcW > 0 && letterbox.srcH > 0) {
                val mappedFit = mapDetectionsToPreviewFitCenter(
                    finalDets, letterbox.srcW, letterbox.srcH, vw, vh
                )
                filterViewArtifacts(mappedFit, vw, vh)
            } else finalDets

            val stable = overlayStabilizer.updateAndStabilize(cleaned)
            overlayView.setDetections(stable)

            if (bestThreat != null) {
                val z = when (bestThreat.zone) {
                    Zone.LEFT -> "LEFT"
                    Zone.MID -> "AHEAD"
                    Zone.RIGHT -> "RIGHT"
                }
                val (ds, _, _) = computeDangerScore(bestThreat.score, bestThreat.areaRatio, bestThreat.zone.name, bestThreat.labelRaw)

                tvLeft.text = "THREAT: ${bestThreat.labelPretty} | $z | DS=${String.format(Locale.US, "%.2f", ds)} | ${bestThreat.prox}"
            } else {
                tvLeft.text = "THREAT: -"
            }

            val mode = if (useDangerScore) "DS" else "CONF"
            tvRight.text = "FPS=${String.format(Locale.US, "%.1f", lastFps)} | pre=${preMs.toInt()} inf=${infMs.toInt()} post=${postMs.toInt()} | dets=${finalDets.size} | $mode"
        }

        // Fix B: Crowd-Modus – aggregierte Warnung bei Menschenmengen
        val personNorm = normLabel("Person")
        val personsInMid = finalDets.count { d ->
            normLabel(d.label) == personNorm &&
                    d.score >= confThreshold &&
                    zoneFor(d.box.centerX(), letterbox.srcW.toFloat()) == Zone.MID &&
                    (max(0f, d.box.width()) * max(0f, d.box.height())) /
                    max(1f, letterbox.srcW.toFloat() * letterbox.srcH.toFloat()) >= 0.020f
        }

        if (personsInMid >= crowdThreshold) {
            maybeSpeakCrowd(personsInMid, preMs, infMs, postMs, finalDets.size)
        } else if (bestThreat != null) {
            // PersistentGate entscheidet ob gewarnt wird (ersetzt Stability-Check + Cooldown)
            maybeSpeakThreat(bestThreat, preMs, infMs, postMs, finalDets.size, decision.threatsCount)
        }

        // Orientierungskanal für Ampel + Zebrastreifen (unabhängig von Gefahrenwarnungen)
        maybeAnnounceOrientation(finalDets, letterbox.srcW.toFloat(), letterbox.srcH.toFloat())
    }

    private fun mapDetectionsToPreviewFitCenter(
        dets: List<Detection>, srcW: Int, srcH: Int, viewW: Float, viewH: Float
    ): List<Detection> {
        val sW = srcW.toFloat().coerceAtLeast(1f)
        val sH = srcH.toFloat().coerceAtLeast(1f)
        val scale = min(viewW / sW, viewH / sH)
        val padX = (viewW - sW * scale) / 2f
        val padY = (viewH - sH * scale) / 2f

        return dets.map { d ->
            val b = d.box
            val r = RectF(
                b.left * scale + padX, b.top * scale + padY,
                b.right * scale + padX, b.bottom * scale + padY
            )
            d.copy(box = r)
        }
    }

    private fun filterViewArtifacts(dets: List<Detection>, viewW: Float, viewH: Float): List<Detection> {
        if (viewW <= 1f || viewH <= 1f) return dets
        val minSize = 6f
        val maxAreaRatio = 0.85f
        val viewArea = viewW * viewH

        return dets.filter { d ->
            val r = d.box
            val w = r.width()
            val h = r.height()
            if (w < minSize || h < minSize) return@filter false
            if (r.right < -2f || r.left > viewW + 2f) return@filter false
            if (r.bottom < -2f || r.top > viewH + 2f) return@filter false
            val area = w * h
            area / viewArea <= maxAreaRatio
        }
    }

    private fun nmsPerClass(dets: List<Detection>, iouThr: Float) =
        dets.groupBy { it.label }.values.flatMap { nms(it, iouThr) }

    private fun nms(dets: List<Detection>, iouThr: Float): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val keep = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            sorted.removeAll { iou(best.box, it.box) >= iouThr }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val x1 = max(a.left, b.left); val y1 = max(a.top, b.top)
        val x2 = min(a.right, b.right); val y2 = min(a.bottom, b.bottom)
        val interW = max(0f, x2 - x1); val interH = max(0f, y2 - y1)
        val inter = interW * interH
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun filterEdgeArtifacts(dets: List<Detection>, srcW: Int, srcH: Int): List<Detection> {
        if (dets.isEmpty()) return dets
        val w = srcW.toFloat(); val h = srcH.toFloat()
        val marginPx = 6f; val minSizePx = 12f
        val frameArea = w * h

        return dets.filter { d ->
            val b = d.box
            val bw = b.width(); val bh = b.height()
            if (bw < minSizePx || bh < minSizePx) return@filter false

            val touchesEdge = b.left <= marginPx || b.top <= marginPx ||
                    b.right >= (w - marginPx) || b.bottom >= (h - marginPx)
            if (!touchesEdge) return@filter true

            val areaRatio = (bw * bh) / frameArea
            d.score >= 0.70f || areaRatio >= 0.010f
        }
    }

    // EINFACHE ORT INIT (kein NNAPI!)
    private fun initOrtSimple(modelBytes: ByteArray) {
        if (ortEnv == null) ortEnv = OrtEnvironment.getEnvironment()
        val env = ortEnv!!

        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        ortSession = env.createSession(modelBytes, opts)
    }

    private fun runOrt(input: OnnxTensor): OrtOutput {
        val session = ortSession ?: throw IllegalStateException("ORT null")
        val result = session.run(mapOf("images" to input))
        return OrtOutput.from(result[0], result)
    }

    private fun decodeYolo(out: OrtOutput, lb: Letterbox): MutableList<Detection> {
        val f = out.data ?: return mutableListOf()
        val dets = mutableListOf<Detection>()

        fun addDet(cx: Float, cy: Float, w: Float, h: Float, cls: Int, score: Float) {
            val rect = lb.unletterbox(cx, cy, w, h)
            dets.add(Detection(rect, score, labelFor(cls)))
        }

        when (f.layout) {
            OrtOutput.FloatTensor.Layout.B_N_C -> {
                for (i in 0 until f.n) {
                    val row = f.getRowBNC(i)
                    val cx = row[0]; val cy = row[1]; val w = row[2]; val h = row[3]
                    var bestClsA = -1; var bestScoreA = 0f
                    for (c in 4 until row.size) {
                        val s = row[c]
                        if (s > bestScoreA) { bestScoreA = s; bestClsA = c - 4 }
                    }
                    if (bestClsA >= 0) addDet(cx, cy, w, h, bestClsA, bestScoreA)
                }
            }
            OrtOutput.FloatTensor.Layout.B_C_N -> {
                for (i in 0 until f.n) {
                    val cx = f.getBCN(0, i); val cy = f.getBCN(1, i)
                    val w = f.getBCN(2, i); val h = f.getBCN(3, i)
                    var bestClsA = -1; var bestScoreA = 0f
                    for (c in 4 until f.c) {
                        val s = f.getBCN(c, i)
                        if (s > bestScoreA) { bestScoreA = s; bestClsA = c - 4 }
                    }
                    if (bestClsA >= 0) addDet(cx, cy, w, h, bestClsA, bestScoreA)
                }
            }
            else -> {}
        }
        return dets
    }

    // ── PersistentGate-basierte Warn-Entscheidung ──────────────────────────
    private fun maybeSpeakThreat(t: Threat, preMs: Float, infMs: Float, postMs: Float,
                                 detCount: Int, threatsCount: Int) {
        val now = SystemClock.elapsedRealtime()

        // DangerScore berechnen (für Gate-Entscheidung + Logging)
        val (danger, classPrio, zoneW) = computeDangerScore(t.score, t.areaRatio, t.zone.name, t.labelRaw)

        // PersistentGate: Event-basierte Entscheidung (NEW / ESCALATE / REMINDER / SUPPRESS)
        val gate = persistGate.decide(t, now, danger)
        if (!gate.allow) return

        // Fix A: Klassen-Cooldown – bei PERSIST_NEW prüfen ob gleiche Klasse kürzlich gewarnt wurde
        // Verhindert Dauerbeschallung in Menschenmengen (Person A → Person B → Person C)
        val labelNorm = normLabel(t.labelRaw)
        val classCooldownMs = when {
            labelNorm == normLabel("Person") && t.zone == Zone.MID -> 6000L
            labelNorm == normLabel("Person") -> 4000L
            else -> 3000L
        }
        if (gate.reason == "PERSIST_NEW" || gate.reason == "PERSIST_INIT") {
            val lastForClass = lastClassWarnMs[labelNorm] ?: 0L
            if (now - lastForClass < classCooldownMs) return
        }

        // Minimaler TTS-Abstand (harte Grenze: keine 2 Ansagen innerhalb 2s)
        if (now - lastTtsAtMs < 2000L) return

        val msg = WarnTextBuilder.buildGerman(t.labelPretty, t.zone, t.prox, t.motionHint)
        lastTtsAtMs = now
        lastClassWarnMs[labelNorm] = now  // Fix A: Klassen-Zeitstempel aktualisieren

        val baseReason = if (useDangerScore) "DANGER_SCORE_V1" else "CONFIDENCE_ONLY"
        val reason = "$baseReason|${gate.reason}"
        warnEngine.logWarnEvent(t.labelRaw, t.labelPretty, t.zone.name, t.prox.name, t.motionHint.name,
            t.score, t.areaRatio, gate.cooldownMs, reason,
            dangerScore = danger, classPriority = classPrio, zoneWeight = zoneW, k = 2.5f)

        tts?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "warn_$now")
    }

    // ── Fix B: Crowd-Modus – aggregierte Warnung bei Menschenmengen ─────
    private fun maybeSpeakCrowd(personCount: Int, preMs: Float, infMs: Float, postMs: Float, detCount: Int) {
        val now = SystemClock.elapsedRealtime()

        // Crowd-Cooldown: nicht öfter als alle 10s
        if (now - lastCrowdWarnMs < crowdCooldownMs) return

        // Globale TTS-Mindestpause
        if (now - lastTtsAtMs < 2000L) return

        val msg = "Mehrere Personen voraus."
        lastCrowdWarnMs = now
        lastTtsAtMs = now
        lastClassWarnMs[normLabel("Person")] = now  // zählt auch als Klassen-Warnung

        val baseReason = if (useDangerScore) "DANGER_SCORE_V1" else "CONFIDENCE_ONLY"
        val reason = "$baseReason|CROWD_MODE_${personCount}"
        warnEngine.logWarnEvent("Person", "Person", "MID", "NEAR", "NONE",
            0f, 0f, crowdCooldownMs, reason,
            dangerScore = 0f, classPriority = getClassPriority("Person"),
            zoneWeight = getZoneWeight("MID"), k = 2.5f)

        tts?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "crowd_$now")
    }

    // ── Fix 7: Orientierungskanal (Ampel, Zebrastreifen) ──────────────────
    // Eigener Cooldown, unabhängig von Gefahrenwarnungen.
    // Nur MID-Zone, nur wenn nah genug. Nicht-unterbrechend (QUEUE_ADD).
    private val orientationLabels = setOf("Ampel", "Zebrastreifen")
    private val orientationAreaMin = mapOf("Ampel" to 0.025f, "Zebrastreifen" to 0.020f)
    private val orientationCooldownMs = 5000L
    private val lastOrientationSpoken = HashMap<String, Long>()

    private fun maybeAnnounceOrientation(dets: List<Detection>, srcW: Float, srcH: Float) {
        val now = SystemClock.elapsedRealtime()
        val frameArea = max(1f, srcW * srcH)

        for (d in dets) {
            if (d.label !in orientationLabels) continue
            if (d.score < 0.35f) continue

            val b = d.box
            val areaRatio = (max(0f, b.width()) * max(0f, b.height())) / frameArea
            val minArea = orientationAreaMin[d.label] ?: 0.025f
            if (areaRatio < minArea) continue

            // Nur MID-Zone (relevant für eigenen Weg)
            val cxNorm = b.centerX() / max(1f, srcW)
            if (cxNorm < 0.30f || cxNorm > 0.70f) continue

            val lastSpoken = lastOrientationSpoken[d.label] ?: 0L
            if (now - lastSpoken < orientationCooldownMs) continue

            // Nicht ansagen wenn gerade eine IMMEDIATE-Warnung läuft (< 1s her)
            if (now - lastTtsAtMs < 1000L) continue

            val msg = when (d.label) {
                "Ampel" -> "Ampel voraus."
                "Zebrastreifen" -> "Zebrastreifen voraus."
                else -> continue
            }

            lastOrientationSpoken[d.label] = now
            // QUEUE_ADD: unterbricht keine aktive Gefahrenwarnung
            tts?.speak(msg, TextToSpeech.QUEUE_ADD, null, "orient_${d.label}_$now")

            // Auch in warn_events loggen (für Evaluation)
            val (danger, classPrio, zoneW) = computeDangerScore(d.score, areaRatio, "MID", d.label)
            warnEngine.logWarnEvent(d.label, prettyLabel(d.label), "MID", "FAR", "NONE",
                d.score, areaRatio, orientationCooldownMs, "ORIENTATION_HINT",
                dangerScore = danger, classPriority = classPrio, zoneWeight = zoneW, k = 2.5f)
        }
    }

    private fun makeInputTensor(src: Bitmap, rotationDeg: Int): Pair<OnnxTensor, Letterbox> {
        val rotated = if (rotationDeg != 0 && rotationDeg % 360 != 0) {
            rotateBitmap(src, rotationDeg)
        } else src

        val (lbBitmap, lb) = letterbox(rotated, inputSize, inputSize)
        val floatBuffer = bitmapToNchwFloat(lbBitmap)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv!!, floatBuffer, shape)
        return Pair(tensor, lb)
    }

    private fun rotateBitmap(bm: Bitmap, deg: Int): Bitmap {
        if (deg == 0 || deg % 360 == 0) return bm
        val m = Matrix().apply { postRotate(deg.toFloat()) }
        val rotated = Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, m, true)
        // Ergebnis in reusable kopieren, um Allokation beim nächsten Frame zu vermeiden
        val w = rotated.width; val h = rotated.height
        val out = reusableRotatedBmp?.takeIf { it.width == w && it.height == h && !it.isRecycled }
            ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { reusableRotatedBmp = it }
        val canvas = Canvas(out)
        canvas.drawBitmap(rotated, 0f, 0f, null)
        if (rotated !== bm) rotated.recycle()
        return out
    }

    private fun letterbox(src: Bitmap, dstW: Int, dstH: Int): Pair<Bitmap, Letterbox> {
        val srcW = src.width.toFloat(); val srcH = src.height.toFloat()
        val r = min(dstW / srcW, dstH / srcH)
        val newW = (srcW * r).toInt(); val newH = (srcH * r).toInt()

        // Wiederverwendbare Bitmaps
        val scaled = reusableScaledBmp?.takeIf { it.width == newW && it.height == newH && !it.isRecycled }
            ?: Bitmap.createBitmap(newW, newH, Bitmap.Config.ARGB_8888).also { reusableScaledBmp = it }
        val scaleCanvas = Canvas(scaled)
        val scaleMatrix = Matrix().apply { setScale(newW / srcW, newH / srcH) }
        scaleCanvas.drawBitmap(src, scaleMatrix, null)

        val out = reusableLetterboxBmp?.takeIf { it.width == dstW && it.height == dstH && !it.isRecycled }
            ?: Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also { reusableLetterboxBmp = it }
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        val padX = (dstW - newW) / 2f; val padY = (dstH - newH) / 2f
        canvas.drawBitmap(scaled, padX, padY, null)

        return Pair(out, Letterbox(src.width, src.height, dstW, dstH, r, padX, padY))
    }

    private fun bitmapToNchwFloat(bm: Bitmap): FloatBuffer {
        val w = bm.width; val h = bm.height
        val total = w * h
        val pixels = reusablePixelArray?.takeIf { it.size == total }
            ?: IntArray(total).also { reusablePixelArray = it }
        bm.getPixels(pixels, 0, w, 0, 0, w, h)

        // FloatArray statt DirectFloatBuffer.put(i,v) — Array-Zugriff ~10x schneller als JNI
        val floatCount = 3 * total
        val arr = reusableNchwArray?.takeIf { it.size == floatCount }
            ?: FloatArray(floatCount).also { reusableNchwArray = it }

        for (i in 0 until total) {
            val p = pixels[i]
            arr[i]              = ((p shr 16) and 0xFF) / 255f
            arr[i + total]      = ((p shr 8) and 0xFF) / 255f
            arr[i + total * 2]  = (p and 0xFF) / 255f
        }

        // Einmaliger Bulk-Copy in DirectBuffer (1 JNI-Call statt 1.228.800)
        val fb = reusableFloatBuffer?.takeIf { it.capacity() == floatCount }
            ?: ByteBuffer.allocateDirect(4 * floatCount)
                .order(ByteOrder.nativeOrder()).asFloatBuffer().also { reusableFloatBuffer = it }
        fb.position(0)
        fb.put(arr)
        fb.rewind()
        return fb
    }

    data class Letterbox(val srcW: Int, val srcH: Int, val dstW: Int, val dstH: Int,
                         val scale: Float, val padX: Float, val padY: Float) {
        fun unletterbox(cx: Float, cy: Float, w: Float, h: Float): RectF {
            val cxPx = if (cx <= 1.5f) cx * dstW else cx
            val cyPx = if (cy <= 1.5f) cy * dstH else cy
            val wPx = if (w <= 1.5f) w * dstW else w
            val hPx = if (h <= 1.5f) h * dstH else h

            val left = cxPx - wPx / 2f; val top = cyPx - hPx / 2f
            val right = cxPx + wPx / 2f; val bottom = cyPx + hPx / 2f

            val l = (left - padX) / scale; val t = (top - padY) / scale
            val r = (right - padX) / scale; val b = (bottom - padY) / scale

            return RectF(
                l.coerceIn(0f, srcW.toFloat()), t.coerceIn(0f, srcH.toFloat()),
                r.coerceIn(0f, srcW.toFloat()), b.coerceIn(0f, srcH.toFloat())
            )
        }
    }

    private class OrtOutput(val data: FloatTensor?, val shapeHint: String) {
        companion object {
            fun from(output: OnnxValue, all: OrtSession.Result): OrtOutput {
                val info = output.info as TensorInfo
                val shapeHint = info.shape.joinToString(prefix = "[", postfix = "]")
                val tensor = output as OnnxTensor
                val ft = FloatTensor.tryWrap(tensor.value)
                return OrtOutput(ft, shapeHint)
            }
        }

        class FloatTensor(
            val layout: Layout, val n: Int, val c: Int,
            private val bnc: Array<FloatArray>?, private val bcn: Array<FloatArray>?
        ) {
            enum class Layout { B_N_C, B_C_N, UNKNOWN }
            fun getRowBNC(i: Int): FloatArray = bnc!![i]
            fun getBCN(ch: Int, i: Int): Float = bcn!![ch][i]

            companion object {
                fun tryWrap(value: Any): FloatTensor? {
                    return try {
                        @Suppress("UNCHECKED_CAST")
                        val arr = value as Array<*>
                        if (arr.isEmpty()) return null
                        val first = arr[0]
                        if (first is Array<*>) {
                            val inner = first
                            if (inner.isEmpty()) return null
                            if (inner[0] is FloatArray) {
                                val rows = inner.map { it as FloatArray }.toTypedArray()
                                return if (rows.size > 1000) {
                                    FloatTensor(Layout.B_N_C, rows.size, rows[0].size, bnc = rows, bcn = null)
                                } else {
                                    FloatTensor(Layout.B_C_N, rows[0].size, rows.size, bnc = null, bcn = rows)
                                }
                            }
                        }
                        null
                    } catch (_: Throwable) { null }
                }
            }
        }
    }
}
