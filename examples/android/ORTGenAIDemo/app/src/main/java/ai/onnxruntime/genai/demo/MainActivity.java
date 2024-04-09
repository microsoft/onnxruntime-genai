package ai.onnxruntime.genai.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;


import ai.onnxruntime.genai.demo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity implements GenAIWrapper.TokenUpdateListener {
    private ActivityMainBinding binding;
    private EditText userMsgEdt;
    private GenAIWrapper genAIWrapper;
    private ImageButton sendMsgIB;
    private String promptQuestion;
    private TextView generatedTV;
    private TextView promptTV;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        sendMsgIB = findViewById(R.id.idIBSend);
        userMsgEdt = findViewById(R.id.idEdtMessage);
        generatedTV = findViewById(R.id.sample_text);
        promptTV = findViewById(R.id.user_text);

        // adding on click listener for send message button.
        sendMsgIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Checking if the message entered
                // by user is empty or not.
                if (userMsgEdt.getText().toString().isEmpty()) {
                    // if the edit text is empty display a toast message.
                    Toast.makeText(MainActivity.this, "Please enter your message..", Toast.LENGTH_SHORT).show();
                    return;
                } else {
                    promptQuestion = userMsgEdt.getText().toString();
                    Log.i("GenAI: prompt question", promptQuestion);
                }
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        promptTV.setText(promptQuestion);
                        performQA(promptQuestion);
                    }
                }).start();
                // Clear Edit Text or prompt question.
                userMsgEdt.setText("");
                generatedTV.setText("");
                promptQuestion="";
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void performQA(String question) {

        // manually upload the model. easiest from Android Studio.
        // Create emulator. Make sure it has at least 8GB of internal storage!
        // Debug app to do initial copy
        // In Device Explorer navigate to /data/data/ai.onnxruntime.genai.demo/files
        // Right-click on the files folder an update the phi-int4-cpu folder.
        File fd = getFilesDir();

        genAIWrapper = new GenAIWrapper(fd.getPath() + "/phi2-int4-cpu-compint8/");
        genAIWrapper.setTokenUpdateListener(this);
        genAIWrapper.run(question);
    }

    @Override
    public void onTokenUpdate(String token) {
        runOnUiThread(() -> {
            // Update and aggregate the generated text and write to text box.
            CharSequence generated = generatedTV.getText();
            generatedTV.setText(generated + token);
            generatedTV.invalidate();
        });
    }
}