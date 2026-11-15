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
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.DenseEmbeddingFloatResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingBitResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingByteResults;
import org.elasticsearch.xpack.core.inference.results.GenericDenseEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.external.http.HttpResult;
import org.elasticsearch.xpack.inference.external.request.OutboundRequest;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIEmbeddingsRequest;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType.toLowerCase;

public class VoyageAIEmbeddingsResponseEntity {
    private static final String VALID_EMBEDDING_TYPES_STRING = supportedEmbeddingTypes();

    private static String supportedEmbeddingTypes() {
        String[] validTypes = new String[] {
            toLowerCase(VoyageAIEmbeddingType.FLOAT),
            toLowerCase(VoyageAIEmbeddingType.INT8),
            toLowerCase(VoyageAIEmbeddingType.BIT) };
        Arrays.sort(validTypes);
        return String.join(", ", validTypes);
    }

    record EmbeddingInt8Result(List<EmbeddingInt8ResultEntry> entries) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<EmbeddingInt8Result, Void> PARSER = new ConstructingObjectParser<>(
            EmbeddingInt8Result.class.getSimpleName(),
            true,
            args -> new EmbeddingInt8Result((List<EmbeddingInt8ResultEntry>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), EmbeddingInt8ResultEntry.PARSER::apply, new ParseField("data"));
        }
    }

    record EmbeddingInt8ResultEntry(Integer index, List<Integer> embedding) {

        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<EmbeddingInt8ResultEntry, Void> PARSER = new ConstructingObjectParser<>(
            EmbeddingInt8ResultEntry.class.getSimpleName(),
            true,
            args -> new EmbeddingInt8ResultEntry((Integer) args[0], (List<Integer>) args[1])
        );

        static {
            PARSER.declareInt(constructorArg(), new ParseField("index"));
            PARSER.declareIntArray(constructorArg(), new ParseField("embedding"));
        }

        private static void checkByteBounds(Integer value) {
            if (value < Byte.MIN_VALUE || value > Byte.MAX_VALUE) {
                throw new IllegalArgumentException(Strings.format("Value [%d] is out of range for a byte", value));
            }
        }

        public DenseEmbeddingByteResults.Embedding toInferenceByteEmbedding() {
            embedding.forEach(EmbeddingInt8ResultEntry::checkByteBounds);
            return DenseEmbeddingByteResults.Embedding.of(embedding.stream().map(Integer::byteValue).toList());
        }
    }

    record EmbeddingFloatResult(List<EmbeddingFloatResultEntry> entries) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<EmbeddingFloatResult, Void> PARSER = new ConstructingObjectParser<>(
            EmbeddingFloatResult.class.getSimpleName(),
            true,
            args -> new EmbeddingFloatResult((List<EmbeddingFloatResultEntry>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), EmbeddingFloatResultEntry.PARSER::apply, new ParseField("data"));
        }
    }

    record EmbeddingFloatResultEntry(Integer index, List<Float> embedding) {

        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<EmbeddingFloatResultEntry, Void> PARSER = new ConstructingObjectParser<>(
            EmbeddingFloatResultEntry.class.getSimpleName(),
            true,
            args -> new EmbeddingFloatResultEntry((Integer) args[0], (List<Float>) args[1])
        );

        static {
            PARSER.declareInt(constructorArg(), new ParseField("index"));
            PARSER.declareFloatArray(constructorArg(), new ParseField("embedding"));
        }

        public DenseEmbeddingFloatResults.Embedding toInferenceFloatEmbedding() {
            return DenseEmbeddingFloatResults.Embedding.of(embedding);
        }
    }

    /**
     * The contextualized embeddings API returns, for each input document, a list of embeddings: one for each chunk of the document.
     */
    record ContextualInt8Document(Integer index, List<EmbeddingInt8ResultEntry> chunks) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualInt8Document, Void> PARSER = new ConstructingObjectParser<>(
            ContextualInt8Document.class.getSimpleName(),
            true,
            args -> new ContextualInt8Document((Integer) args[0], (List<EmbeddingInt8ResultEntry>) args[1])
        );

        static {
            PARSER.declareInt(constructorArg(), new ParseField("index"));
            PARSER.declareObjectArray(constructorArg(), EmbeddingInt8ResultEntry.PARSER::apply, new ParseField("data"));
        }
    }

    record ContextualInt8Result(List<ContextualInt8Document> documents) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualInt8Result, Void> PARSER = new ConstructingObjectParser<>(
            ContextualInt8Result.class.getSimpleName(),
            true,
            args -> new ContextualInt8Result((List<ContextualInt8Document>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), ContextualInt8Document.PARSER::apply, new ParseField("data"));
        }
    }

    record ContextualFloatDocument(Integer index, List<EmbeddingFloatResultEntry> chunks) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualFloatDocument, Void> PARSER = new ConstructingObjectParser<>(
            ContextualFloatDocument.class.getSimpleName(),
            true,
            args -> new ContextualFloatDocument((Integer) args[0], (List<EmbeddingFloatResultEntry>) args[1])
        );

        static {
            PARSER.declareInt(constructorArg(), new ParseField("index"));
            PARSER.declareObjectArray(constructorArg(), EmbeddingFloatResultEntry.PARSER::apply, new ParseField("data"));
        }
    }

    record ContextualFloatResult(List<ContextualFloatDocument> documents) {
        @SuppressWarnings("unchecked")
        public static final ConstructingObjectParser<ContextualFloatResult, Void> PARSER = new ConstructingObjectParser<>(
            ContextualFloatResult.class.getSimpleName(),
            true,
            args -> new ContextualFloatResult((List<ContextualFloatDocument>) args[0])
        );

        static {
            PARSER.declareObjectArray(constructorArg(), ContextualFloatDocument.PARSER::apply, new ParseField("data"));
        }
    }

    /**
     * Parses the response of the Voyage AI contextualized embeddings API. A response looks like:
     *
     * <pre>
     * <code>
     * {
     *  "object": "list",
     *  "data": [
     *      {
     *          "object": "list",
     *          "data": [
     *              { "object": "embedding", "embedding": [ ... ], "index": 0, "text": "Sample text 1" }
     *          ],
     *          "index": 0
     *      }
     *  ],
     *  "model": "voyage-context-4",
     *  "usage": { "total_tokens": 10 }
     * }
     * </code>
     * </pre>
     *
     * The inference API returns one embedding for each input, so the response must hold exactly one chunk for each input document.
     * Inputs sent to the inference API are already chunked and are far smaller than the chunk size sent to Voyage AI, so Voyage AI is not
     * expected to split them further.
     */
    private static InferenceServiceResults parseContextualized(XContentParser jsonParser, VoyageAIEmbeddingType embeddingType)
        throws IOException {
        switch (embeddingType) {
            case FLOAT -> {
                var result = ContextualFloatResult.PARSER.apply(jsonParser, null);
                var chunks = result.documents.stream().map(d -> singleChunk(d.index, d.chunks)).toList();
                return new DenseEmbeddingFloatResults(chunks.stream().map(EmbeddingFloatResultEntry::toInferenceFloatEmbedding).toList());
            }
            case INT8 -> {
                var result = ContextualInt8Result.PARSER.apply(jsonParser, null);
                var chunks = result.documents.stream().map(d -> singleChunk(d.index, d.chunks)).toList();
                return new DenseEmbeddingByteResults(chunks.stream().map(EmbeddingInt8ResultEntry::toInferenceByteEmbedding).toList());
            }
            case BIT, BINARY -> {
                var result = ContextualInt8Result.PARSER.apply(jsonParser, null);
                var chunks = result.documents.stream().map(d -> singleChunk(d.index, d.chunks)).toList();
                return new DenseEmbeddingBitResults(chunks.stream().map(EmbeddingInt8ResultEntry::toInferenceByteEmbedding).toList());
            }
            default -> throw new IllegalArgumentException(
                Strings.format("Illegal embedding_type value: %s. Supported types are: %s", embeddingType, VALID_EMBEDDING_TYPES_STRING)
            );
        }
    }

    private static <T> T singleChunk(Integer documentIndex, List<T> chunks) {
        if (chunks.size() != 1) {
            throw new IllegalStateException(
                Strings.format(
                    "Expected one embedding for the input at index [%s] but the contextualized embeddings API returned [%d]",
                    documentIndex,
                    chunks.size()
                )
            );
        }
        return chunks.get(0);
    }

    /**
     * Parses the VoyageAI json response.
     * For a request like:
     *
     * <pre>
     *     <code>
     *        {
     *          "input": [
     *            "Sample text 1",
     *            "Sample text 2"
     *          ],
     *          "model": "voyage-3-large"
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
     *          "object": "embedding",
     *          "embedding": [
     *              -0.009327292,
     *              -0.0028842222,
     *          ],
     *          "index": 0
     *      },
     *      {
     *          "object": "embedding",
     *          "embedding": [ ... ],
     *          "index": 1
     *      }
     *  ],
     *  "model": "voyage-3-large",
     *  "usage": {
     *      "total_tokens": 10
     *  }
     * }
     * </code>
     * </pre>
     */
    public static InferenceServiceResults fromResponse(OutboundRequest outboundRequest, HttpResult response) throws IOException {
        var parserConfig = XContentParserConfiguration.EMPTY.withDeprecationHandler(LoggingDeprecationHandler.INSTANCE);
        var embeddingsRequest = (VoyageAIEmbeddingsRequest) outboundRequest;
        var serviceSettings = embeddingsRequest.getServiceSettings();
        var embeddingType = serviceSettings.embeddingType();
        // The embedding task returns results in a generic format, as it is not tied to the type of the embeddings
        var isEmbeddingTask = embeddingsRequest.getTaskType() == TaskType.EMBEDDING;

        try (XContentParser jsonParser = XContentFactory.xContent(XContentType.JSON).createParser(parserConfig, response.body())) {
            if (serviceSettings.isContextual()) {
                return parseContextualized(jsonParser, embeddingType);
            }
            switch (embeddingType) {
                case FLOAT -> {
                    var embeddingResult = EmbeddingFloatResult.PARSER.apply(jsonParser, null);

                    List<DenseEmbeddingFloatResults.Embedding> embeddingList = embeddingResult.entries.stream()
                        .map(EmbeddingFloatResultEntry::toInferenceFloatEmbedding)
                        .toList();
                    return isEmbeddingTask
                        ? new GenericDenseEmbeddingFloatResults(embeddingList)
                        : new DenseEmbeddingFloatResults(embeddingList);
                }
                case INT8 -> {
                    var embeddingResult = EmbeddingInt8Result.PARSER.apply(jsonParser, null);
                    List<DenseEmbeddingByteResults.Embedding> embeddingList = embeddingResult.entries.stream()
                        .map(EmbeddingInt8ResultEntry::toInferenceByteEmbedding)
                        .toList();
                    return isEmbeddingTask
                        ? new GenericDenseEmbeddingByteResults(embeddingList)
                        : new DenseEmbeddingByteResults(embeddingList);
                }
                case BIT, BINARY -> {
                    var embeddingResult = EmbeddingInt8Result.PARSER.apply(jsonParser, null);
                    List<DenseEmbeddingByteResults.Embedding> embeddingList = embeddingResult.entries.stream()
                        .map(EmbeddingInt8ResultEntry::toInferenceByteEmbedding)
                        .toList();
                    return isEmbeddingTask
                        ? new GenericDenseEmbeddingBitResults(embeddingList)
                        : new DenseEmbeddingBitResults(embeddingList);
                }
                default -> throw new IllegalArgumentException(
                    Strings.format("Illegal embedding_type value: %s. Supported types are: %s", embeddingType, VALID_EMBEDDING_TYPES_STRING)
                );
            }
        }
    }

    private VoyageAIEmbeddingsResponseEntity() {}
}
