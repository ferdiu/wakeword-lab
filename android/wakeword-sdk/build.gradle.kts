plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.wakeword.lab"
    compileSdk = 36
    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.2")
    testImplementation("junit:junit:4.13.2")
}
