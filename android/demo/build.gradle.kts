plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.wakeword.demo"
    compileSdk = 36
    defaultConfig {
        applicationId = "dev.wakeword.demo"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }
}

dependencies { implementation(project(":wakeword-sdk")) }
