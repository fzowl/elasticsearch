/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.response;

import org.apache.hc.core5.http.HttpResponse;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.XContentParseException;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResults;
import org.elasticsearch.xpack.core.inference.results.EmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.InputTypeTests;
import org.elasticsearch.xpack.inference.external.http.HttpResult;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIEmbeddingsRequest;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelTests.createContextualModel;
import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelTests.createModel;
import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelTests.createMultimodalModel;
import static org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIEmbeddingsRequestTests.createRequest;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.mockito.Mockito.mock;

public class VoyageAIEmbeddingsResponseEntityTests extends ESTestCase {
    public void testFromResponse_CreatesResultsForASingleItem() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          0.014539449,
                          -0.015288644
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        InferenceServiceResults parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 0.014539449F, -0.015288644F })))
        );
    }

    public void testFromResponse_CreatesResultsForMultipleItems() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          0.014539449,
                          -0.015288644
                      ]
                  },
                  {
                      "object": "embedding",
                      "index": 1,
                      "embedding": [
                          0.0123,
                          -0.0123
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        InferenceServiceResults parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(
                List.of(
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.014539449F, -0.015288644F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.0123F, -0.0123F })
                )
            )
        );
    }

    public void testFromResponse_FailsWhenDataFieldIsNotPresent() {
        String responseJson = """
            {
              "object": "list",
              "not_data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          0.014539449,
                          -0.015288644
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        var thrownException = expectThrows(
            java.lang.IllegalArgumentException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                request,
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(thrownException.getMessage(), is("Required [data]"));
    }

    public void testFromResponse_FailsWhenDataFieldNotAnArray() {
        String responseJson = """
            {
              "object": "list",
              "data": {
                  "test": {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          0.014539449,
                          -0.015288644
                      ]
                  }
              },
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        var thrownException = expectThrows(
            XContentParseException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                request,
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(thrownException.getMessage(), containsString("[EmbeddingFloatResult] failed to parse field [data]"));
    }

    public void testFromResponse_FailsWhenEmbeddingsDoesNotExist() {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embeddingzzz": [
                          0.014539449,
                          -0.015288644
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        var thrownException = expectThrows(
            XContentParseException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                request,
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(thrownException.getMessage(), containsString("[EmbeddingFloatResult] failed to parse field [data]"));
    }

    public void testFromResponse_FailsWhenEmbeddingValueIsAString() {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          "abc"
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        var thrownException = expectThrows(
            XContentParseException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                request,
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(thrownException.getMessage(), is("[8:15] [EmbeddingFloatResult] failed to parse field [data]"));
    }

    public void testFromResponse_SucceedsWhenEmbeddingValueIsInt() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          1
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        InferenceServiceResults parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 1.0F })))
        );
    }

    public void testFromResponse_SucceedsWhenEmbeddingValueIsLong() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          40294967295
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        InferenceServiceResults parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(List.of(new DenseEmbeddingFloatResults.Embedding(new float[] { 4.0294965E10F })))
        );
    }

    public void testFromResponse_FailsWhenEmbeddingValueIsAnObject() {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "embedding",
                      "index": 0,
                      "embedding": [
                          {}
                      ]
                  }
              ],
              "model": "voyage-3-large",
              "usage": {
                  "total_tokens": 8
              }
            }
            """;

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        var thrownException = expectThrows(
            XContentParseException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                request,
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(thrownException.getMessage(), is("[8:15] [EmbeddingFloatResult] failed to parse field [data]"));
    }

    public void testFieldsInDifferentOrderServer() throws IOException {
        // The fields of the objects in the data array are reordered
        String response = """
            {
                "object": "list",
                "model": "voyage-3-large",
                "data": [
                    {
                        "embedding": [
                            -0.9,
                            0.5,
                            0.3
                        ],
                        "index": 0,
                        "object": "embedding"
                    },
                    {
                        "index": 0,
                        "embedding": [
                            0.1,
                            0.5
                        ],
                        "object": "embedding"
                    },
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [
                            0.5,
                            0.5
                        ]
                    }
                ],
                "usage": {
                    "total_tokens": 0
                }
            }""";

        VoyageAIEmbeddingsRequest request = createRequest(
            List.of("abc", "def"),
            InputTypeTests.randomSearchAndIngestWithNull(),
            createModel("url", "api_key", null, "voyage-3-large")
        );

        InferenceServiceResults parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), response.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(parsedResults, instanceOf(DenseEmbeddingFloatResults.class));

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(
                List.of(
                    new DenseEmbeddingFloatResults.Embedding(new float[] { -0.9F, 0.5F, 0.3F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.1F, 0.5F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.5F, 0.5F })
                )
            )
        );
    }

    public void testFromResponse_Contextualized_CreatesOneEmbeddingForEachInput() throws IOException {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "list",
                      "index": 0,
                      "data": [ { "object": "embedding", "index": 0, "embedding": [ 0.1, 0.2 ], "text": "abc" } ]
                  },
                  {
                      "object": "list",
                      "index": 1,
                      "data": [ { "object": "embedding", "index": 0, "embedding": [ 0.3, 0.4 ], "text": "def" } ]
                  }
              ],
              "model": "voyage-context-4",
              "usage": { "total_tokens": 8 }
            }
            """;

        var parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            createRequest(List.of("abc", "def"), InputType.INGEST, createContextualModel("url", "api_key", "voyage-context-4")),
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(
            ((DenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(
                List.of(
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.1F, 0.2F }),
                    new DenseEmbeddingFloatResults.Embedding(new float[] { 0.3F, 0.4F })
                )
            )
        );
    }

    public void testFromResponse_Contextualized_FailsWhenAnInputIsSplitIntoSeveralChunks() {
        String responseJson = """
            {
              "object": "list",
              "data": [
                  {
                      "object": "list",
                      "index": 0,
                      "data": [
                          { "object": "embedding", "index": 0, "embedding": [ 0.1, 0.2 ], "text": "a" },
                          { "object": "embedding", "index": 1, "embedding": [ 0.3, 0.4 ], "text": "b" }
                      ]
                  }
              ],
              "model": "voyage-context-4"
            }
            """;

        var thrownException = expectThrows(
            IllegalStateException.class,
            () -> VoyageAIEmbeddingsResponseEntity.fromResponse(
                createRequest(List.of("ab"), InputType.INGEST, createContextualModel("url", "api_key", "voyage-context-4")),
                new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
            )
        );

        assertThat(
            thrownException.getMessage(),
            is("Expected one embedding for the input at index [0] but the contextualized embeddings API returned [2]")
        );
    }

    private static final String MULTIMODAL_FLOAT_RESPONSE = """
        {
          "object": "list",
          "data": [
              { "object": "embedding", "embedding": [ 0.1, -0.2 ], "index": 0 },
              { "object": "embedding", "embedding": [ 0.3, 0.4 ], "index": 1 }
          ],
          "model": "voyage-multimodal-3.5",
          "usage": { "text_tokens": 5, "image_pixels": 2000000, "total_tokens": 3 }
        }
        """;

    public void testFromResponse_EmbeddingTask_Float_CreatesGenericResultsWithTheEmbeddings() throws IOException {
        var request = createRequest(
            List.of("abc", "def"),
            InputType.INGEST,
            createMultimodalModel("url", "api_key", VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS, null, null, "voyage-multimodal-3.5")
        );

        var parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), MULTIMODAL_FLOAT_RESPONSE.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(parsedResults, instanceOf(GenericDenseEmbeddingFloatResults.class));
        assertThat(
            ((GenericDenseEmbeddingFloatResults) parsedResults).embeddings(),
            is(
                List.of(
                    new GenericDenseEmbeddingFloatResults.Embedding(new float[] { 0.1F, -0.2F }),
                    new GenericDenseEmbeddingFloatResults.Embedding(new float[] { 0.3F, 0.4F })
                )
            )
        );
    }

    public void testFromResponse_EmbeddingTask_Int8_CreatesGenericByteResultsWithTheEmbeddings() throws IOException {
        var responseJson = """
            {
              "object": "list",
              "data": [
                  { "object": "embedding", "embedding": [ 1, -2, 127 ], "index": 0 },
                  { "object": "embedding", "embedding": [ -128, 4, 5 ], "index": 1 }
              ],
              "model": "voyage-multimodal-3.5"
            }
            """;
        var request = createRequest(
            List.of("abc", "def"),
            InputType.INGEST,
            createMultimodalModel(
                "url",
                "api_key",
                VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
                null,
                VoyageAIEmbeddingType.INT8,
                "voyage-multimodal-3.5"
            )
        );

        var parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(parsedResults, instanceOf(GenericDenseEmbeddingByteResults.class));
        assertThat(
            ((GenericDenseEmbeddingByteResults) parsedResults).embeddings(),
            is(
                List.of(
                    new GenericDenseEmbeddingByteResults.Embedding(new byte[] { 1, -2, 127 }),
                    new GenericDenseEmbeddingByteResults.Embedding(new byte[] { -128, 4, 5 })
                )
            )
        );
    }

    public void testFromResponse_EmbeddingTask_Binary_CreatesGenericBitResultsWithTheEmbeddings() throws IOException {
        var responseJson = """
            {
              "object": "list",
              "data": [
                  { "object": "embedding", "embedding": [ 10, -20 ], "index": 0 },
                  { "object": "embedding", "embedding": [ 30, 40 ], "index": 1 }
              ],
              "model": "voyage-multimodal-3.5"
            }
            """;
        var request = createRequest(
            List.of("abc", "def"),
            InputType.INGEST,
            createMultimodalModel(
                "url",
                "api_key",
                VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
                null,
                randomFrom(VoyageAIEmbeddingType.BIT, VoyageAIEmbeddingType.BINARY),
                "voyage-multimodal-3.5"
            )
        );

        var parsedResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            request,
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );

        assertThat(parsedResults, instanceOf(GenericDenseEmbeddingBitResults.class));
        assertThat(
            ((GenericDenseEmbeddingBitResults) parsedResults).embeddings(),
            is(
                List.of(
                    new EmbeddingByteResults.Embedding(new byte[] { 10, -20 }),
                    new EmbeddingByteResults.Embedding(new byte[] { 30, 40 })
                )
            )
        );
    }

    public void testFromResponse_TextEmbeddingTask_Int8AndBinary_CreateDenseResultsWithTheEmbeddings() throws IOException {
        var responseJson = """
            {
              "object": "list",
              "data": [ { "object": "embedding", "embedding": [ 10, -20 ], "index": 0 } ],
              "model": "voyage-3-large"
            }
            """;

        var int8Results = VoyageAIEmbeddingsResponseEntity.fromResponse(
            createRequest(
                List.of("abc"),
                InputType.INGEST,
                createModel(
                    "url",
                    "api_key",
                    VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
                    null,
                    null,
                    "voyage-3-large",
                    VoyageAIEmbeddingType.INT8
                )
            ),
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );
        assertThat(int8Results, instanceOf(DenseEmbeddingByteResults.class));
        assertThat(
            ((DenseEmbeddingByteResults) int8Results).embeddings(),
            is(List.of(new DenseEmbeddingByteResults.Embedding(new byte[] { 10, -20 })))
        );

        var bitResults = VoyageAIEmbeddingsResponseEntity.fromResponse(
            createRequest(
                List.of("abc"),
                InputType.INGEST,
                createModel(
                    "url",
                    "api_key",
                    VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
                    null,
                    null,
                    "voyage-3-large",
                    VoyageAIEmbeddingType.BINARY
                )
            ),
            new HttpResult(mock(HttpResponse.class), responseJson.getBytes(StandardCharsets.UTF_8))
        );
        assertThat(bitResults, instanceOf(DenseEmbeddingBitResults.class));
        assertThat(
            ((DenseEmbeddingBitResults) bitResults).embeddings(),
            is(List.of(new EmbeddingByteResults.Embedding(new byte[] { 10, -20 })))
        );
    }
}
