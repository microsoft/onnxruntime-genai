package ai.onnxruntime.genai.example.javavalidator

import ai.onnxruntime.genai.*

import android.os.Build;
import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.*
import org.junit.runner.RunWith
import java.io.IOException
import java.util.*

private const val TAG = "ORTGenAIAndroidTest"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    val activityTestRule = ActivityScenarioRule(MainActivity::class.java)

    @Before
    fun Start() {
        Log.println(Log.INFO, TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @After
    fun TearDown() {
    }

    @Test
    fun runBasicTest() {
        // the test model requires manual input as the token ids have to be < 1000 but the configured tokenizer
        // has a larger vocab size and the input ids it generates are not valid.
        val model = Model("test_model")
        val params = model.createGeneratorParams()

        val sequenceLength = 4
        val batchSize = 2
        val tokenIds: IntArray = intArrayOf(0, 0, 0, 52,
                                            0, 0, 195, 731)

        val maxLength = 10
        params.setInput(tokenIds, sequenceLength, batchSize)
        params.setSearchOption("max_length", maxLength.toDouble())

        val outputSequences = model.generate(params)

        val expectedOutput =
            intArrayOf(
                0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                0, 0, 195, 731, 731, 114, 114, 114, 114, 114
            )

        for (i in 0 until batchSize) {
            val outputIds: IntArray = outputSequences.getSequence(i.toLong())
            for (j in 0 until maxLength) {
                Assert.assertEquals(outputIds[j], expectedOutput[i * maxLength + j])
            }
        }
    }
}
