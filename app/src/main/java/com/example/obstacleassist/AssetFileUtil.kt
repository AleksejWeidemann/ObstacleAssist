package com.example.obstacleassist

import android.content.Context
import java.io.File

object AssetFileUtil {
    /** Kopiert eine Asset-Datei in den App-Speicher und gibt den absoluten Pfad zurück. */
    fun assetFilePath(context: Context, assetName: String): String {
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
        context.assets.open(assetName).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile.absolutePath
    }
}
