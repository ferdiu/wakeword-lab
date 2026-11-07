plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// The wake word under which takes are recorded, injected at build time, e.g.
// ./gradlew -PwakeWord="Hey Vox" :recorder:assembleDebug. Backslashes and quotes
// are escaped so arbitrary phrases survive the buildConfigField embedding.
val wakeWord: String = (project.findProperty("wakeWord") as? String) ?: "Hey Vox"
val escapedWakeWord = wakeWord.replace("\\", "\\\\").replace("\"", "\\\"")

android {
    namespace = "dev.wakeword.recorder"
    compileSdk = 36
    defaultConfig {
        applicationId = "dev.wakeword.recorder"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("String", "WAKE_WORD", "\"$escapedWakeWord\"")
    }
    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}
