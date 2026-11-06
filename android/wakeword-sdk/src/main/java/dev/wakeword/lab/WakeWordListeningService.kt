package dev.wakeword.lab

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/** Optional user-visible microphone foreground service. Start it from a visible Activity. */
class WakeWordListeningService : Service() {
    private var detector: WakeWordDetector? = null
    private lateinit var notificationManager: NotificationManager

    override fun onCreate() {
        super.onCreate()
        active = this
        notificationManager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            notificationManager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Wake-word listening", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (detector != null) return START_NOT_STICKY
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            stopSelf()
            return START_NOT_STICKY
        }
        // Starting from an Activity in the foreground satisfies microphone FGS while-in-use rules.
        val notification = buildNotification("Preparing offline wake-word model…")
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (error: Exception) {
            // Android 14+ enforces while-in-use permission and foreground-start eligibility.
            // Do not crash if the host starts this service outside an allowed foreground flow.
            stopSelf()
            return START_NOT_STICKY
        }
        detector = WakeWordDetector(this, object : WakeWordDetector.Listener {
            override fun onDetected(score: Float) {
                notificationManager.notify(NOTIFICATION_ID, buildNotification("Wake word detected (${"%.2f".format(score)})"))
                uiListener?.invoke("wake word detected, score=${"%.3f".format(score)}")
            }

            override fun onScores(maxScore: Float, threshold: Float) {
                uiListener?.invoke("heartbeat: maxScore=${"%.3f".format(maxScore)} threshold=$threshold")
            }

            override fun onError(error: Throwable) {
                uiListener?.invoke("detector error: ${error.message ?: error.javaClass.simpleName}")
                notificationManager.notify(NOTIFICATION_ID, buildNotification("Wake-word listener stopped: ${error.message ?: error.javaClass.simpleName}"))
                stopSelf()
            }
        })
        // A threshold chosen in the UI before the service existed takes effect on this run.
        pendingThresholdOverride?.let { detector?.thresholdOverride = it }
        try {
            detector?.start()
        } catch (error: Throwable) {
            notificationManager.notify(NOTIFICATION_ID, buildNotification("Cannot start: ${error.message}"))
            stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (active === this) active = null
        detector?.close()
        detector = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    /** Live threshold override; forwarded to the detector while it is running. */
    fun setThreshold(value: Float?) {
        detector?.thresholdOverride = value
    }

    private fun buildNotification(message: String): Notification {
        val stopIntent = Intent(this, WakeWordListeningService::class.java).setAction(ACTION_STOP)
        val stopPending = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        return builder.setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Wake-word detector active")
            .setContentText(message)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(android.R.drawable.ic_media_pause, "Stop", stopPending).build())
            .build()
    }

    companion object {
        const val ACTION_STOP = "dev.wakeword.lab.action.STOP"
        private const val CHANNEL_ID = "wakeword-listening"
        private const val NOTIFICATION_ID = 8401

        /** Receives detector log lines (heartbeat, detections, errors); always on the main thread. */
        @Volatile var uiListener: ((String) -> Unit)? = null
        @Volatile var active: WakeWordListeningService? = null
        /** Threshold chosen before the service starts; applied when the detector is created. */
        @Volatile var pendingThresholdOverride: Float? = null
    }
}
