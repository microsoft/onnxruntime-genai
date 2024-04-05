package ai.onnxruntime.genai.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;


import org.w3c.dom.Text;

import java.io.File;
import java.util.concurrent.CountDownLatch;

import ai.onnxruntime.genai.demo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity implements GenAIWrapper.TokenUpdateListener {
    private ActivityMainBinding binding;
    private GenAIWrapper genAIWrapper;
    private ImageButton sendMsgIB;

    private String promptQuestion = "What is the square root of pi.";

    private TextView generatedTV;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        sendMsgIB = findViewById(R.id.idIBSend);
        generatedTV = findViewById(R.id.sample_text);

        // adding on click listener for send message button.
        sendMsgIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // TODO:
                // Checking if the message entered
                // by user is empty or not.
                /* if (userMsgEdt.getText().toString().isEmpty()) {
                // if the edit text is empty display a toast message.
                    Toast.makeText(MainActivity.this, "Please enter your message..", Toast.LENGTH_SHORT).show();
                    return;
                 }

                 // Calling a method to send message
                 // to our GenAI bot to get response.

                 // sendMessage(userMsgEdt.getText().toString());

                 // below line we are setting text in our edit text as empty
                 userMsgEdt.setText("");
 */
                  performQA();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void performQA() {

        // manually upload the model. easiest from Android Studio.
        // Create emulator. Make sure it has at least 8GB of internal storage!
        // Debug app to do initial copy
        // In Device Explorer navigate to /data/data/ai.onnxruntime.genai.demo/files
        // Right-click on the files folder an update the phi-int4-cpu folder.
        File fd = getFilesDir();

        genAIWrapper = new GenAIWrapper(fd.getPath() + "/phi2-int4-cpu-compint8/");
        genAIWrapper.setTokenUpdateListener(this);
        String output = genAIWrapper.run(promptQuestion);
//        generatedTV.setText(output);
    }

    @Override
    public void onTokenUpdate(String token) {
        runOnUiThread(() -> {
            CharSequence generated = generatedTV.getText();
            generatedTV.setText(generated + token);
        });

    }
}