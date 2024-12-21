package ai.onnxruntime.genai.example.javavalidator

import ai.onnxruntime.genai.*
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.*
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*


private const val TAG = "ORTGenAIAndroidTest"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    val activityTestRule = ActivityScenarioRule(MainActivity::class.java)

    @Before
    fun start() {
        Log.i(TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @Throws(IOException::class)
    private fun copyModelFromAssets(): String {
        // NOTE: We have to read from the app's assets (app/src/main/assets) and write the the app's filesDir.
        // The unit test's context cannot be used. you'll get mysterious errors like File.mkdirs() will return false and
        // assertManager.open() throws even if the filename is valid if you try and use it.
        // Test context is InstrumentationRegistry.getInstrumentation().targetContext.
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val assetManager = context.assets
        val files: Array<String>?
        try {
            files = context.assets.list("model")
        } catch (e: IOException) {
            Log.e("copyModelFromAssets", "Failed to find `model` folder in app assets.", e)
            throw e
        }

        val filesDir = context.filesDir
        if (!filesDir.exists()) {
            throw IOException("Files directory is not valid: " + filesDir.absolutePath)
        }

        val modelTargetPath = filesDir.absolutePath + File.separator + "model"
        val modelTargetDir = File(modelTargetPath)
        if (!modelTargetDir.exists() and !modelTargetDir.mkdirs()) {
            throw IOException("Target directory could not be created: " + modelTargetDir.absolutePath)
        }

        // the model data is expected to be large, so use a decent sized buffer
        val buffer = ByteArray(64*1024)

        for (filename in files!!) {
            try {
                val outFile = File(modelTargetPath + File.separator + filename)
                if (!outFile.exists()) {
                    outFile.createNewFile()
                }

                val srcStream = assetManager.open("model/$filename")
                val dstStream = FileOutputStream(outFile)
                var bytesRead: Int
                while (srcStream.read(buffer).also { bytesRead = it } != -1) {
                    dstStream.write(buffer, 0, bytesRead)
                }

                srcStream.close()
                dstStream.flush()
                dstStream.close()
            } catch (e: IOException) {
                Log.e("copyModelFromAssets", "Failed to copy file from assets/model: $filename", e)
            }
        }

        return modelTargetPath
    }

    @Test
    fun runBasicTest() {
        // We have to copy the model directory from the app assets to the app's files directory so it can be accessed
        // normally by the C++ GenAI code. when it's in the apk it is compressed.
        val newModelPath = copyModelFromAssets()

        // the test model requires manual input as the token ids have to be < 1000 but the configured tokenizer
        // has a larger vocab size and the input ids it generates are not valid.
        val model = Model(newModelPath)
        val params = GeneratorParams(model)

        val sequenceLength = 4
        val batchSize = 2
        val tokenIds: IntArray = intArrayOf(0, 0, 0, 52,
                                            0, 0, 195, 731)

        val maxLength = 10
        params.setSearchOption("max_length", maxLength.toDouble())
        params.setSearchOption("batch_size", batchSize.toDouble())

        val generator = Generator(model, params)
        generator.appendTokens(tokenIds)
        while(!generator.isDone()) {
            generator.generateNextToken()
        }

        val expectedOutput =
            intArrayOf(
                0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                0, 0, 195, 731, 731, 114, 114, 114, 114, 114
            )

        for (i in 0 until batchSize) {
            val outputIds: IntArray = generator.getSequence(i.toLong())
            for (j in 0 until maxLength) {
                Assert.assertEquals(outputIds[j], expectedOutput[i * maxLength + j])
            }
        }

        Log.i("runBasicTest", "GenAI output matched expected data.")
    }
}
