/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

public class NonGenerativeTest {
  @Test
  public void requestValidationAndStructuredConversion() {
    assertThrows(IllegalArgumentException.class, () -> new StructuredQuestion("", "prompt"));
    assertThrows(NullPointerException.class, () -> new StructuredRequest(null, null));
    assertThrows(
        IllegalArgumentException.class,
        () -> new StructuredRequest(new BigInteger("1"), new LinkedHashMap<>()));
    assertThrows(
        IllegalArgumentException.class,
        () -> new StructuredQuestion("noul", new BigDecimal("1.25")));

    Map<String, Object> cycle = new LinkedHashMap<>();
    cycle.put("self", cycle);
    assertThrows(
        IllegalArgumentException.class, () -> new StructuredRequest(cycle, new LinkedHashMap<>()));

    List<Object> root = new ArrayList<>();
    List<Object> current = root;
    for (int i = 0; i < 128; ++i) {
      List<Object> child = new ArrayList<>();
      current.add(child);
      current = child;
    }
    assertThrows(
        IllegalArgumentException.class, () -> new StructuredRequest(root, new LinkedHashMap<>()));

    Map<String, Object> answer = new LinkedHashMap<>();
    answer.put("id", "q");
    answer.put("type", "noul");
    answer.put("noul", Double.valueOf(0.75));
    answer.put("choice", null);
    answer.put("score", null);
    answer.put("confidence", Double.valueOf(0.9));
    answer.put("probabilities", new LinkedHashMap<String, Double>());
    answer.put("legend", new LinkedHashMap<String, String>());
    Map<String, Object> result = new LinkedHashMap<>();
    result.put("model", "synthetic");
    result.put("answers", Arrays.asList(answer));

    ModelResult converted = new ModelResult(result);
    assertEquals("synthetic", converted.model);
    assertEquals("q", converted.answers.get(0).id);
    assertEquals(0.75, converted.answers.get(0).noul.doubleValue());
  }

  @Test
  public void optInClmPackageCoversLifecycleCacheAndResults() throws GenAIException {
    String path = System.getenv("ORTGENAI_TEST_CLM_PACKAGE");
    if (path == null || path.isEmpty()) {
      return;
    }
    Map<String, StructuredQuestion> questions = new LinkedHashMap<>();
    questions.put("q", new StructuredQuestion("noul", "Is this suitable?"));
    Map<String, Object> state = new LinkedHashMap<>();
    state.put("weather", "rain");
    try (RankingSession session = new RankingSession(path, "cpu")) {
      session.setCacheCapacity(2, 1024 * 1024);
      ModelResult result = session.run(new StructuredRequest(state, questions));
      assertNotNull(result.model);
      assertEquals(1, result.answers.size());
      Map<String, Object> candidates = new LinkedHashMap<>();
      candidates.put("inside", Arrays.asList("museum", Boolean.TRUE));
      candidates.put("outside", "picnic");
      assertEquals(
          2,
          session
              .rank(new FreeFormRankRequest(state, "Choose the best activity", candidates))
              .items
              .size());
      assertEquals(2, session.getCacheStats().entryCapacity);
      session.clearCache();
      assertEquals(0, session.getCacheStats().entries);
      session.invalidateCache();
    }

    final RankingSession concurrent = new RankingSession(path, "cpu");
    ExecutorService executor = Executors.newFixedThreadPool(3);
    CountDownLatch start = new CountDownLatch(1);
    Future<?> operation =
        executor.submit(
            () -> {
              start.await();
              try {
                concurrent.run(new StructuredRequest(state, questions));
              } catch (IllegalStateException expected) {
                // close won the monitor
              }
              return null;
            });
    Future<?> closeOne =
        executor.submit(
            () -> {
              start.await();
              concurrent.close();
              return null;
            });
    Future<?> closeTwo =
        executor.submit(
            () -> {
              start.await();
              concurrent.close();
              return null;
            });
    start.countDown();
    try {
      operation.get();
      closeOne.get();
      closeTwo.get();
    } catch (Exception error) {
      throw new AssertionError(error);
    } finally {
      concurrent.close();
      executor.shutdownNow();
    }
  }

  @Test
  public void optInKevPackageCoversDecisionAndDirectoryTokenizer() throws GenAIException {
    String path = System.getenv("ORTGENAI_TEST_KEV_PACKAGE");
    if (path == null || path.isEmpty()) {
      return;
    }
    Map<String, StructuredQuestion> questions = new LinkedHashMap<>();
    questions.put("q", new StructuredQuestion("noul", "Take an umbrella?"));
    try (DecisionSession session = new DecisionSession(path);
        DirectoryTokenizer tokenizer = new DirectoryTokenizer(path)) {
      assertTrue(tokenizer.encode("rain").length > 0);
      assertTrue(session.getPrefixReuseEnabled());
      assertTrue(!session.getPrefixReuseStatus().isEmpty());
      session.setPrefixReuseEnabled(false);
      assertTrue(!session.getPrefixReuseEnabled());
      session.setPrefixReuseEnabled(true);
      session.setPrefixCacheCapacity(1, 1024 * 1024);
      assertEquals(1, session.getPrefixCacheStats().entryCapacity);
      assertNotNull(session.decide(new StructuredRequest("rain", questions)).model);
      PrefixReuseStats prefixStats = session.getPrefixReuseStats();
      assertTrue(prefixStats.prefixRuns + prefixStats.branchRuns + prefixStats.fallbackRuns > 0);
      session.clearCache();
      assertEquals(0, session.getPrefixCacheStats().entries);

      Map<String, Object> mutated = new LinkedHashMap<>();
      StructuredRequest cyclicRequest = new StructuredRequest(mutated, questions);
      mutated.put("self", mutated);
      assertThrows(GenAIException.class, () -> session.decide(cyclicRequest));
    }

    final DecisionSession decision = new DecisionSession(path);
    final DirectoryTokenizer tokenizer = new DirectoryTokenizer(path);
    ExecutorService executor = Executors.newFixedThreadPool(4);
    CountDownLatch start = new CountDownLatch(1);
    Future<?> decide =
        executor.submit(
            () -> {
              start.await();
              try {
                decision.decide(new StructuredRequest("rain", questions));
              } catch (IllegalStateException expected) {
                // close won the monitor
              }
              return null;
            });
    Future<?> encode =
        executor.submit(
            () -> {
              start.await();
              try {
                tokenizer.encode("rain");
              } catch (IllegalStateException expected) {
                // close won the monitor
              }
              return null;
            });
    Future<?> closeDecision =
        executor.submit(
            () -> {
              start.await();
              decision.close();
              decision.close();
              return null;
            });
    Future<?> closeTokenizer =
        executor.submit(
            () -> {
              start.await();
              tokenizer.close();
              tokenizer.close();
              return null;
            });
    start.countDown();
    try {
      decide.get();
      encode.get();
      closeDecision.get();
      closeTokenizer.get();
    } catch (Exception error) {
      throw new AssertionError(error);
    } finally {
      decision.close();
      tokenizer.close();
      executor.shutdownNow();
    }
  }
}
