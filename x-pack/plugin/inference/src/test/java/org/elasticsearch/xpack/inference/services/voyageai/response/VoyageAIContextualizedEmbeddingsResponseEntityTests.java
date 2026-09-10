/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.response;

import org.apache.http.HttpResponse;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.InputTypeTests;
import org.elasticsearch.xpack.inference.external.http.HttpResult;
import org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIContextualizedEmbeddingsRequest;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelTests.createModel;
import static org.hamcrest.Matchers.is;
import static org.mockito.Mockito.mock;

public class VoyageAIContextualizedEmbeddingsResponseEntityTests extends ESTestCase {

    /**
     * The contextualized chunk embeddings response nests one embedding per chunk inside a per-document {@code data}
     * array. The parser must flatten these back into a single ordered list.
     */
    public void testFromResponse_FlattensChunkEmbeddingsAcrossDocuments() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "list",
                      "index": 0,
                      "data": [
                          { "object": "embedding", "index": 0, "embedding": [0.014539449, -0.015288644] },
                          { "object": "embedding", "index": 1, "embedding": [0.0123, -0.0123] }
                      ]
                  },
                  {
                      "object": "list",
                      "index": 1,
                      "data": [
                          { "object": "embedding", "index": 0, "embedding": [0.5, -0.5] }
                      ]
                  }
              ],
              "model": "voyage-context-3",
              "usage": { "total_tokens": 24 }
            }
            """;

        VoyageAIContextualizedEmbeddingsRequest request = new VoyageAIContextualizedEmbeddingsRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-context-3")
        );

        InferenceServiceResults parsedResults = VoyageAIContextualizedEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(
                List.of(
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.014539449F, -0.015288644F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.0123F, -0.0123F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.5F, -0.5F })
                )
            )
        );
    }

    public void testFromResponse_CreatesResultsForASingleChunk() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "list",
                      "index": 0,
                      "data": [
                          { "object": "embedding", "index": 0, "embedding": [0.014539449, -0.015288644] }
                      ]
                  }
              ],
              "model": "voyage-context-3",
              "usage": { "total_tokens": 8 }
            }
            """;

        VoyageAIContextualizedEmbeddingsRequest request = new VoyageAIContextualizedEmbeddingsRequest(
            List.of("abc"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-context-3")
        );

        InferenceServiceResults parsedResults = VoyageAIContextualizedEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.014539449F, -0.015288644F })))
        );
    }
}
