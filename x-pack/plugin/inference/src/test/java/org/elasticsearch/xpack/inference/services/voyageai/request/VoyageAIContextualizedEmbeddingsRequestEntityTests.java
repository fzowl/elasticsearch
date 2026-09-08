/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.Matchers.is;

public class VoyageAIContextualizedEmbeddingsRequestEntityTests extends ESTestCase {

    private static final String TEST_MODEL_ID = "voyage-context-3";
    private static final int TEST_RATE_LIMIT = 10;
    private static final int TEST_DIMENSIONS = 2048;
    private static final int TEST_MAX_INPUT_TOKENS = 1024;

    private static VoyageAIEmbeddingsServiceSettings serviceSettings(VoyageAIEmbeddingType embeddingType) {
        return new VoyageAIEmbeddingsServiceSettings(
            TEST_MODEL_ID,
            new RateLimitSettings(TEST_RATE_LIMIT),
            embeddingType,
            SimilarityMeasure.DOT_PRODUCT,
            TEST_DIMENSIONS,
            TEST_MAX_INPUT_TOKENS,
            false
        );
    }

    public void testXContent_WritesNestedInputs_ForMultipleDocuments() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        var entity = new VoyageAIContextualizedEmbeddingsRequestEntity(
            List.of(List.of("doc 1 chunk 1", "doc 1 chunk 2"), List.of("doc 2 chunk 1")),
            false,
            InputType.INTERNAL_SEARCH,
            serviceSettings(embeddingType),
            new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null)
        );

        assertJson(entity, Strings.format("""
            {
                "inputs": [["doc 1 chunk 1", "doc 1 chunk 2"], ["doc 2 chunk 1"]],
                "model": "%s",
                "input_type": "query",
                "output_dimension": %d,
                "output_dtype": "%s"
            }
            """, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString()));
    }

    public void testXContent_WritesFlatInputs_ForSingleDocument_WhenFlattening() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        var entity = VoyageAIContextualizedEmbeddingsRequestEntity.fromSingleDocument(
            List.of("chunk 1", "chunk 2"),
            true,
            InputType.INGEST,
            serviceSettings(embeddingType),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS
        );

        assertJson(entity, Strings.format("""
            {
                "inputs": ["chunk 1", "chunk 2"],
                "model": "%s",
                "input_type": "document",
                "output_dimension": %d,
                "output_dtype": "%s"
            }
            """, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString()));
    }

    public void testXContent_WritesNestedInputs_ForSingleDocument_WhenNotFlattening() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        var entity = VoyageAIContextualizedEmbeddingsRequestEntity.fromSingleDocument(
            List.of("chunk 1", "chunk 2"),
            false,
            InputType.INGEST,
            serviceSettings(embeddingType),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS
        );

        assertJson(entity, Strings.format("""
            {
                "inputs": [["chunk 1", "chunk 2"]],
                "model": "%s",
                "input_type": "document",
                "output_dimension": %d,
                "output_dtype": "%s"
            }
            """, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString()));
    }

    private static void assertJson(VoyageAIContextualizedEmbeddingsRequestEntity entity, String expected) throws IOException {
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        assertThat(Strings.toString(builder), is(XContentHelper.stripWhitespace(expected)));
    }
}
