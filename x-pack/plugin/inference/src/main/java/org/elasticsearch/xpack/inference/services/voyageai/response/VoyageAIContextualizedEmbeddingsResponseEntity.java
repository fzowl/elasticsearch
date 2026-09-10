/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.response;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.xcontent.LoggingDeprecationHandler;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.external.http.HttpResult;
import org.elasticsearch.xpack.inference.external.request.OutboundRequest;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIContextualizedEmbeddingsRequest;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType.toLowerCase;

/**
 * Parses responses from VoyageAI's contextualized chunk embeddings API
 * (<a href="https://docs.voyageai.com/docs/contextualized-chunk-embeddings">docs</a>).
 * <p>
 * Unlike the standard embeddings response, this response nests embeddings one level deeper: the top-level {@code data}
 * array contains one entry per input document, and each of those entries carries its own {@code data} array with one
 * embedding per chunk. This parser flattens the per-document chunk embeddings back into a single ordered list, matching
 * the order in which the chunks were sent.
 */
public class VoyageAIContextualizedEmbeddingsResponseEntity {
    private static final String VALID_EMBEDDING_TYPES_STRING = supportedEmbeddingTypes();

    private static String supportedEmbeddingTypes() {
        String[] validTypes = new String[] {
            toLowerCase(VoyageAIEmbeddingType.FLOAT),
            toLowerCase(VoyageAIEmbeddingType.INT8),
            toLowerCase(VoyageAIEmbeddingType.BIT) };
        Arrays.sort(validTypes);
        return String.join(", ", validTypes);
    }

    // ---- int8 / bit ----

    record ContextualizedInt8Result(List<ContextualizedInt8Document> documents) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualizedInt8Result, Void> PARSER = new ConstructingObjectParser<>(
            ContextualizedInt8Result.class.getSimpleName(),
            true,
            args -> new ContextualizedInt8Result((List<ContextualizedInt8Document>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), ContextualizedInt8Document.PARSER::apply, new ParseField("data"));
        }
    }

    record ContextualizedInt8Document(List<VoyageAIEmbeddingsResponseEntity.EmbeddingInt8ResultEntry> chunks) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualizedInt8Document, Void> PARSER = new ConstructingObjectParser<>(
            ContextualizedInt8Document.class.getSimpleName(),
            true,
            args -> new ContextualizedInt8Document((List<VoyageAIEmbeddingsResponseEntity.EmbeddingInt8ResultEntry>) args[0])
        );

        static {
            PARSER.declareObjectArray(
                constructorArg(),
                VoyageAIEmbeddingsResponseEntity.EmbeddingInt8ResultEntry.PARSER::apply,
                new ParseField("data")
            );
        }
    }

    // ---- float ----

    record ContextualizedFloatResult(List<ContextualizedFloatDocument> documents) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualizedFloatResult, Void> PARSER = new ConstructingObjectParser<>(
            ContextualizedFloatResult.class.getSimpleName(),
            true,
            args -> new ContextualizedFloatResult((List<ContextualizedFloatDocument>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), ContextualizedFloatDocument.PARSER::apply, new ParseField("data"));
        }
    }

    record ContextualizedFloatDocument(List<VoyageAIEmbeddingsResponseEntity.EmbeddingFloatResultEntry> chunks) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualizedFloatDocument, Void> PARSER = new ConstructingObjectParser<>(
            ContextualizedFloatDocument.class.getSimpleName(),
            true,
            args -> new ContextualizedFloatDocument((List<VoyageAIEmbeddingsResponseEntity.EmbeddingFloatResultEntry>) args[0])
        );

        static {
            PARSER.declareObjectArray(
                constructorArg(),
                VoyageAIEmbeddingsResponseEntity.EmbeddingFloatResultEntry.PARSER::apply,
                new ParseField("data")
            );
        }
    }

    /**
     * Parses the VoyageAI contextualized chunk embeddings json response. For a request like:
     *
     * <pre>
     *     <code>
     *        {
     *          "inputs": [["Sample chunk 1", "Sample chunk 2"]],
     *          "model": "voyage-context-3"
     *        }
     *     </code>
     * </pre>
     *
     * The response would look like:
     *
     * <pre>
     * <code>
     * {
     *  "object": "list",
     *  "data": [
     *      {
     *          "object": "list",
     *          "index": 0,
     *          "data": [
     *              { "object": "embedding", "embedding": [ ... ], "index": 0 },
     *              { "object": "embedding", "embedding": [ ... ], "index": 1 }
     *          ]
     *      }
     *  ],
     *  "model": "voyage-context-3",
     *  "usage": { "total_tokens": 10 }
     * }
     * </code>
     * </pre>
     */
    public static InferenceServiceResults fromResponse(OutboundRequest outboundRequest, HttpResult response) throws IOException {
        var parserConfig = XContentParserConfiguration.EMPTY.withDeprecationHandler(LoggingDeprecationHandler.INSTANCE);
        var embeddingType = ((VoyageAIContextualizedEmbeddingsRequest) outboundRequest).getServiceSettings().embeddingType();

        try (XContentParser jsonParser = XContentFactory.xContent(XContentType.JSON).createParser(parserConfig, response.body())) {
            switch (embeddingType) {
                case FLOAT -> {
                    var result = ContextualizedFloatResult.PARSER.apply(jsonParser, null);
                    List<DenseEmbeddingFloatResults.Embedding> embeddingList = result.documents.stream()
                        .flatMap(document -> document.chunks.stream())
                        .map(VoyageAIEmbeddingsResponseEntity.EmbeddingFloatResultEntry::toInferenceFloatEmbedding)
                        .toList();
                    return new DenseEmbeddingFloatResults(embeddingList);
                }
                case INT8 -> {
                    var result = ContextualizedInt8Result.PARSER.apply(jsonParser, null);
                    List<DenseEmbeddingByteResults.Embedding> embeddingList = result.documents.stream()
                        .flatMap(document -> document.chunks.stream())
                        .map(VoyageAIEmbeddingsResponseEntity.EmbeddingInt8ResultEntry::toInferenceByteEmbedding)
                        .toList();
                    return new DenseEmbeddingByteResults(embeddingList);
                }
                case BIT, BINARY -> {
                    var result = ContextualizedInt8Result.PARSER.apply(jsonParser, null);
                    List<DenseEmbeddingByteResults.Embedding> embeddingList = result.documents.stream()
                        .flatMap(document -> document.chunks.stream())
                        .map(VoyageAIEmbeddingsResponseEntity.EmbeddingInt8ResultEntry::toInferenceByteEmbedding)
                        .toList();
                    return new DenseEmbeddingBitResults(embeddingList);
                }
                default -> throw new IllegalArgumentException(
                    Strings.format("Illegal embedding_type value: %s. Supported types are: %s", embeddingType, VALID_EMBEDDING_TYPES_STRING)
                );
            }
        }
    }

    private VoyageAIContextualizedEmbeddingsResponseEntity() {}
}
