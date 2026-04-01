package com.example.myapplication2.yolo

import android.graphics.RectF

data class Detection(
    val box: RectF, // normalized [0,1] in preview coordinates (after rotation handling)
    val score: Float,
    val classId: Int,
    val className: String,
    val group: Group,
    val labelZh: String
) {
    enum class Group {
        PERSON,
        VEHICLE,
        OBSTACLE
    }
}

