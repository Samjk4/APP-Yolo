package com.example.myapplication2.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class YoloV8TfliteDetector(
    private val context: Context,
    private val modelAssetName: String = "yolov8n_float16.tflite",
    private val inputSize: Int = 640,
) : AutoCloseable {
    private val interpreter: Interpreter

    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true)
        }
        interpreter = Interpreter(loadMappedFile(context, modelAssetName), opts)
    }

    fun detect(image: ImageProxy): List<Detection> {
        val bmp = rgba8888ImageProxyToBitmap(image)
        val rotationDegrees = image.imageInfo.rotationDegrees
        val rotated = if (rotationDegrees != 0) rotateBitmap(bmp, rotationDegrees) else bmp

        val inShape = interpreter.getInputTensor(0).shape() // [1,640,640,3] or [1,3,640,640]
        require(inShape.size == 4 && inShape[0] == 1) { "Unexpected input shape: ${inShape.contentToString()}" }
        val isNhwc = inShape[3] == 3
        val inH = if (isNhwc) inShape[1] else inShape[2]
        val inW = if (isNhwc) inShape[2] else inShape[3]

        val letterbox = letterbox(rotated, inW, inH)
        val inputBuffer = bitmapToInputBuffer(letterbox.bitmap, isNhwc)

        val outputTensor = interpreter.getOutputTensor(0)
        val outShape = outputTensor.shape() // e.g. [1, 84, 8400] or [1, 8400, 84]
        require(outShape.size == 3 && outShape[0] == 1) { "Unexpected output shape: ${outShape.contentToString()}" }
        val d0 = outShape[1]
        val d1 = outShape[2]

        // We'll read raw floats into a flat array, then provide accessor (c,i) -> value.
        val outFloats = FloatArray(d0 * d1)
        val outBuffer = ByteBuffer.allocateDirect(4 * outFloats.size).order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outBuffer)

        outBuffer.rewind()
        outBuffer.asFloatBuffer().get(outFloats)

        val (numBoxes, numChannels, get) = if (d0 <= 200 && d1 >= 1000) {
            // [84, 8400]
            val channels = d0
            val boxes = d1
            Triple(boxes, channels) { c: Int, i: Int -> outFloats[c * boxes + i] }
        } else {
            // [8400, 84]
            val boxes = d0
            val channels = d1
            Triple(boxes, channels) { c: Int, i: Int -> outFloats[i * channels + c] }
        }

        val candidates = outputToCandidates(
            numBoxes = numBoxes,
            numChannels = numChannels,
            get = get,
            letterbox = letterbox,
            rotatedW = rotated.width,
            rotatedH = rotated.height
        )

        val picked = nms(candidates, DEFAULT_IOU_THRESHOLD)
        return picked.map { cand ->
            val className = COCO80[cand.classId].lowercase()
            val group = when (className) {
                "person" -> Detection.Group.PERSON
                "car", "bus", "truck", "motorcycle", "bicycle" -> Detection.Group.VEHICLE
                else -> Detection.Group.OBSTACLE
            }
            val labelZh = when (group) {
                Detection.Group.PERSON -> "人"
                Detection.Group.VEHICLE -> "車"
                Detection.Group.OBSTACLE -> "障礙物"
            }
            Detection(
                box = cand.box,
                score = cand.score,
                classId = cand.classId,
                className = className,
                group = group,
                labelZh = labelZh
            )
        }
    }

    override fun close() {
        interpreter.close()
    }

    private data class Candidate(
        val box: RectF,
        val score: Float,
        val classId: Int
    )

    private data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val inW: Int,
        val inH: Int,
        val outW: Int,
        val outH: Int
    ) {
        fun mapRectToOriginal(r: RectF): RectF {
            val left = (r.left - padX) / scale
            val top = (r.top - padY) / scale
            val right = (r.right - padX) / scale
            val bottom = (r.bottom - padY) / scale
            return RectF(left, top, right, bottom)
        }
    }

    private fun letterbox(src: Bitmap, dstW: Int, dstH: Int): LetterboxResult {
        val inW = src.width
        val inH = src.height
        val scale = min(dstW.toFloat() / inW.toFloat(), dstH.toFloat() / inH.toFloat())
        val newW = (inW * scale).toInt()
        val newH = (inH * scale).toInt()
        val padX = (dstW - newW) / 2f
        val padY = (dstH - newH) / 2f

        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(resized, padX, padY, null)
        return LetterboxResult(out, scale, padX, padY, inW, inH, dstW, dstH)
    }

    private fun bitmapToInputBuffer(bmp: Bitmap, isNhwc: Boolean): ByteBuffer {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        val out = ByteBuffer.allocateDirect(4 * 3 * w * h).order(ByteOrder.nativeOrder())
        if (isNhwc) {
            // TensorFlow Lite YOLO models are commonly NHWC.
            for (p in pixels) {
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                out.putFloat(r)
                out.putFloat(g)
                out.putFloat(b)
            }
        } else {
            // Fallback for NCHW exported models.
            val rPlane = FloatArray(w * h)
            val gPlane = FloatArray(w * h)
            val bPlane = FloatArray(w * h)
            for (i in pixels.indices) {
                val p = pixels[i]
                rPlane[i] = ((p shr 16) and 0xFF) / 255f
                gPlane[i] = ((p shr 8) and 0xFF) / 255f
                bPlane[i] = (p and 0xFF) / 255f
            }
            for (v in rPlane) out.putFloat(v)
            for (v in gPlane) out.putFloat(v)
            for (v in bPlane) out.putFloat(v)
        }
        out.rewind()
        return out
    }

    private fun normalizeBox(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        letterbox: LetterboxResult,
        rotatedW: Int,
        rotatedH: Int
    ): RectF {
        // Some exports output boxes in input-pixel space, some in normalized [0,1].
        // We auto-detect and convert both.
        val looksNormalized = x1 in -0.1f..1.1f && y1 in -0.1f..1.1f && x2 in -0.1f..1.1f && y2 in -0.1f..1.1f
        val lx1 = if (looksNormalized) x1 * letterbox.outW else x1
        val ly1 = if (looksNormalized) y1 * letterbox.outH else y1
        val lx2 = if (looksNormalized) x2 * letterbox.outW else x2
        val ly2 = if (looksNormalized) y2 * letterbox.outH else y2
        val mapped = letterbox.mapRectToOriginal(RectF(lx1, ly1, lx2, ly2))
        return RectF(
            (mapped.left / rotatedW).coerceIn(0f, 1f),
            (mapped.top / rotatedH).coerceIn(0f, 1f),
            (mapped.right / rotatedW).coerceIn(0f, 1f),
            (mapped.bottom / rotatedH).coerceIn(0f, 1f)
        )
    }

    private fun decodeCandidates(
        numBoxes: Int,
        numClasses: Int,
        get: (Int, Int) -> Float,
        letterbox: LetterboxResult,
        rotatedW: Int,
        rotatedH: Int
    ): List<Candidate> {
        val candidates = ArrayList<Candidate>(256)
        for (i in 0 until numBoxes) {
            val cx = get(0, i)
            val cy = get(1, i)
            val w = get(2, i)
            val h = get(3, i)

            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val rawScore = get(4 + c, i)
                val s = if (rawScore < 0f || rawScore > 1f) sigmoid(rawScore) else rawScore
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c
                }
            }
            if (bestScore < DEFAULT_CONF_THRESHOLD) continue
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f
            val norm = normalizeBox(x1, y1, x2, y2, letterbox, rotatedW, rotatedH)
            candidates.add(Candidate(norm, bestScore, bestClass))
        }
        return candidates
    }

    private fun decodeCandidatesWithObj(
        numBoxes: Int,
        numClasses: Int,
        get: (Int, Int) -> Float,
        letterbox: LetterboxResult,
        rotatedW: Int,
        rotatedH: Int
    ): List<Candidate> {
        val candidates = ArrayList<Candidate>(256)
        for (i in 0 until numBoxes) {
            val cx = get(0, i)
            val cy = get(1, i)
            val w = get(2, i)
            val h = get(3, i)
            val rawObj = get(4, i)
            val obj = if (rawObj < 0f || rawObj > 1f) sigmoid(rawObj) else rawObj

            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val rawScore = get(5 + c, i)
                val cls = if (rawScore < 0f || rawScore > 1f) sigmoid(rawScore) else rawScore
                val s = obj * cls
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c
                }
            }
            if (bestScore < DEFAULT_CONF_THRESHOLD) continue
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f
            val norm = normalizeBox(x1, y1, x2, y2, letterbox, rotatedW, rotatedH)
            candidates.add(Candidate(norm, bestScore, bestClass))
        }
        return candidates
    }

    private fun outputToCandidates(
        numBoxes: Int,
        numChannels: Int,
        get: (Int, Int) -> Float,
        letterbox: LetterboxResult,
        rotatedW: Int,
        rotatedH: Int
    ): List<Candidate> {
        // Support both:
        // - YOLOv8 style: [x,y,w,h,80cls] => 84 channels
        // - YOLOv5 style: [x,y,w,h,obj,80cls] => 85 channels
        return if (numChannels >= 85) {
            val numClasses = numChannels - 5
            decodeCandidatesWithObj(numBoxes, numClasses, get, letterbox, rotatedW, rotatedH)
        } else {
            val numClasses = numChannels - 4
            decodeCandidates(numBoxes, numClasses, get, letterbox, rotatedW, rotatedH)
        }
    }

    private fun rgba8888ImageProxyToBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.rewind()

        val w = image.width
        val h = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        if (pixelStride == 4 && rowStride == w * 4) {
            out.copyPixelsFromBuffer(buffer)
            return out
        }

        val rgba = ByteArray(rowStride * h)
        buffer.get(rgba)
        val argb = IntArray(w * h)
        var dst = 0
        for (y in 0 until h) {
            var src = y * rowStride
            for (x in 0 until w) {
                val r = rgba[src].toInt() and 0xFF
                val g = rgba[src + 1].toInt() and 0xFF
                val b = rgba[src + 2].toInt() and 0xFF
                val a = rgba[src + 3].toInt() and 0xFF
                argb[dst++] = (a shl 24) or (r shl 16) or (g shl 8) or b
                src += pixelStride
            }
        }
        out.setPixels(argb, 0, w, 0, 0, w, h)
        return out
    }

    private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        val m = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun nms(cands: List<Candidate>, iouThreshold: Float): List<Candidate> {
        val sorted = cands.sortedByDescending { it.score }
        val picked = ArrayList<Candidate>(sorted.size)
        val removed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (removed[i]) continue
            val a = sorted[i]
            picked.add(a)
            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                val b = sorted[j]
                if (a.classId != b.classId) continue
                if (iou(a.box, b.box) > iouThreshold) removed[j] = true
            }
        }
        return picked
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x)))

    private fun loadMappedFile(context: Context, assetName: String): ByteBuffer {
        context.assets.openFd(assetName).use { afd ->
            FileInputStream(afd.fileDescriptor).use { fis ->
                val channel = fis.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }

    companion object {
        private const val DEFAULT_CONF_THRESHOLD = 0.25f
        private const val DEFAULT_IOU_THRESHOLD = 0.45f

        private val COCO80 = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}

