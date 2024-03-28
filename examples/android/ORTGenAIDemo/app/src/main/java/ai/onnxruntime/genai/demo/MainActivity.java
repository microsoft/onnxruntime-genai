package ai.onnxruntime.genai.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import java.io.File;

import ai.onnxruntime.genai.demo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private GenAIWrapper genAIWrapper;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // manually upload the model. easiest from Android Studio.
        // Create emulator. Make sure it has at least 8GB of internal storage!
        // Debug app to do initial copy
        // In Device Explorer navigate to /data/data/ai.onnxruntime.genai.demo/files
        // Right-click on the files folder an update the phi-int4-cpu folder.
        File fd = getFilesDir();
        genAIWrapper = new GenAIWrapper(fd.getPath() + "/phi2-int4-cpu");
        String output = genAIWrapper.run("What is the square root of pi.");

        TextView tv = binding.sampleText;
        tv.setText(output);
    }

}