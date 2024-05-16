package ai.onnxruntime.genai.example.javavalidator

import ai.onnxruntime.genai.*

import android.os.Build;
import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.microsoft.appcenter.espresso.Factory
import com.microsoft.appcenter.espresso.ReportHelper
import org.junit.*
import org.junit.runner.RunWith
import java.io.IOException
import java.util.*

private const val TAG = "ORTGenAIAndroidTest"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    val activityTestRule = ActivityScenarioRule(MainActivity::class.java)

    @get:Rule
    var reportHelper: ReportHelper = Factory.getReportHelper()

    @Before
    fun Start() {
        reportHelper.label("Starting App")
        Log.println(Log.INFO, TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @After
    fun TearDown() {
        reportHelper.label("Stopping App")
    }

    @Test
    fun runBasicTest() {
        // TODO: Add tests
    }
}
