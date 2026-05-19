package com.example.obstacleassist

import android.content.Context
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.util.Locale

class CsvLogger(
    context: Context,
    private val fileName: String,
    private val header: String
) {
    private val lock = Any()
    private val file: File = File(context.filesDir, fileName)
    private var writer: BufferedWriter? = null

    fun open() {
        synchronized(lock) {
            if (writer != null) return
            val exists = file.exists()
            writer = BufferedWriter(OutputStreamWriter(FileOutputStream(file, true), Charsets.UTF_8))
            if (!exists || file.length() == 0L) {
                writer!!.write(header)
                writer!!.write("\n")
                writer!!.flush()
            }
        }
    }

    fun close() {
        synchronized(lock) {
            writer?.flush()
            writer?.close()
            writer = null
        }
    }

    fun logRow(vararg values: Any?) {
        val line = buildLine(values.toList())
        synchronized(lock) {
            if (writer == null) open()
            writer!!.write(line)
            writer!!.write("\n")
        }
    }

    fun flush() {
        synchronized(lock) { writer?.flush() }
    }

    private fun buildLine(values: List<Any?>): String {
        return values.joinToString(separator = ";") { v ->
            when (v) {
                null -> ""
                is Float -> String.format(Locale.US, "%.4f", v)
                is Double -> String.format(Locale.US, "%.4f", v)
                else -> escape(v.toString())
            }
        }
    }

    private fun escape(s: String): String {
        val needsQuote = s.contains(';') || s.contains('"') || s.contains('\n') || s.contains('\r')
        if (!needsQuote) return s
        val escaped = s.replace("\"", "\"\"")
        return "\"$escaped\""
    }
}
