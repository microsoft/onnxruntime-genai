package ai.onnxruntime.genai;

import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestPlan;

public class GenAITestExecutionListener implements TestExecutionListener {
  public void testPlanExecutionFinished(TestPlan testPlan) {
    GenAI.shutdown();
  }
}
