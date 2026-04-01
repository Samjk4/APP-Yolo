package com.example.myapplication2

import android.Manifest
import android.content.Context
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.myapplication2.yolo.Detection
import com.example.myapplication2.yolo.YoloV8TfliteDetector
import java.util.concurrent.Executors

@Composable
fun YoloCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    var hasCameraPermission by remember { mutableStateOf(false) }
    var cameraStatus by remember { mutableStateOf("初始化中…") }
    val permissionText = if (hasCameraPermission) "已允許" else "未允許"
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    LaunchedEffect(Unit) {
        permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        val previewView = remember {
            PreviewView(context).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        }

        val detectionsState: MutableState<List<Detection>> = remember { mutableStateOf(emptyList()) }
        val detectorState = remember { mutableStateOf<YoloV8TfliteDetector?>(null) }

        DisposableEffect(Unit) {
            val detector = YoloV8TfliteDetector(context)
            detectorState.value = detector
            onDispose {
                detector.close()
                detectorState.value = null
                analysisExecutor.shutdown()
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { previewView }
            )

            DetectionOverlay(
                detections = detectionsState.value,
                modifier = Modifier.fillMaxSize()
            )

            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.TopCenter
            ) {
                Column(
                    modifier = Modifier.padding(top = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "DEBUG：YOLO 偵測畫面已載入",
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "權限=$permissionText / 相機=$cameraStatus / 框=${detectionsState.value.size}",
                        color = Color.White,
                        fontSize = 12.sp
                    )
                    Text(
                        text = "YOLO：人 / 車 / 障礙物",
                        color = Color.White,
                        fontSize = 12.sp
                    )
                }
            }
        }

        LaunchedEffect(previewView, lifecycleOwner, hasCameraPermission) {
            if (!hasCameraPermission) {
                cameraStatus = "等待相機權限…"
                return@LaunchedEffect
            }
            try {
                cameraStatus = "綁定中…"
                bindCameraUseCases(
                    context = context,
                    lifecycleOwner = lifecycleOwner,
                    previewView = previewView,
                    detectorState = detectorState,
                    detectionsState = detectionsState,
                    mainExecutor = mainExecutor,
                    analysisExecutor = analysisExecutor
                )
                cameraStatus = "已綁定"
            } catch (t: Throwable) {
                cameraStatus = "失敗：${t.javaClass.simpleName}"
            }
        }
    }
}

private fun bindCameraUseCases(
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    detectorState: MutableState<YoloV8TfliteDetector?>,
    detectionsState: MutableState<List<Detection>>,
    mainExecutor: java.util.concurrent.Executor,
    analysisExecutor: java.util.concurrent.Executor,
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    cameraProviderFuture.addListener(
        {
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(analysisExecutor) { image ->
                val detector = detectorState.value
                if (detector == null) {
                    image.close()
                    return@setAnalyzer
                }

                try {
                    val results = detector.detect(image)
                    mainExecutor.execute { detectionsState.value = results }
                } catch (_: Throwable) {
                    // ignore per-frame failures
                } finally {
                    image.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
        },
        ContextCompat.getMainExecutor(context)
    )
}

@Composable
private fun DetectionOverlay(
    detections: List<Detection>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height

        detections.forEach { det ->
            val r = det.box
            val left = (r.left * w).coerceIn(0f, w)
            val top = (r.top * h).coerceIn(0f, h)
            val right = (r.right * w).coerceIn(0f, w)
            val bottom = (r.bottom * h).coerceIn(0f, h)

            val color = when (det.group) {
                Detection.Group.PERSON -> Color(0xFF00E5FF)
                Detection.Group.VEHICLE -> Color(0xFFFFEA00)
                Detection.Group.OBSTACLE -> Color(0xFFFF1744)
            }

            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = ComposeSize(right - left, bottom - top),
                style = Stroke(width = 4f)
            )

            drawContext.canvas.nativeCanvas.apply {
                val label = "${det.labelZh} ${(det.score * 100f).toInt()}%"
                val paint = android.graphics.Paint().apply {
                    this.color = android.graphics.Color.WHITE
                    this.textSize = 38f
                    this.isAntiAlias = true
                }
                drawText(label, left + 6f, (top - 10f).coerceAtLeast(40f), paint)
            }
        }
    }
}

