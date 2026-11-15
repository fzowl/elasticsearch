/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.DataFormat;
import org.elasticsearch.inference.InferenceString;
import org.elasticsearch.inference.InferenceStringGroup;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.BaseVoyageAIEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.inference.InferenceStringGroup.toStringList;
import static org.elasticsearch.inference.InputType.invalidInputTypeMessage;

/**
 * The request body of a Voyage AI embeddings request. The format of the body is chosen by the service settings, never by the model id.
 * <p>
 * For the embeddings endpoint (neither multimodal nor contextual):
 * <pre><code>
 *     "input": ["first text input", "second text input"]
 * </code></pre>
 * For the contextualized embeddings endpoint, where each text input is a document that Voyage AI chunks when the input type is document:
 * <pre><code>
 *     "inputs": ["first document", "second document"],
 *     "enable_auto_chunking": true,
 *     "chunk_size": 32000
 * </code></pre>
 * For the multimodal embeddings endpoint, where the content of each input is a list of text and base64 image items that are used to
 * generate a single embedding:
 * <pre><code>
 *     "inputs": [
 *       {
 *         "content": [
 *           {"type": "text", "text": "text input"},
 *           {"type": "image_base64", "image_base64": "data:image/jpeg;base64,..."}
 *         ]
 *       }
 *     ]
 * </code></pre>
 *
 * @param input the inputs to generate embeddings for
 * @param inputType the {@link InputType} of the request, which takes precedence over the one of the task settings, may be null
 * @param serviceSettings the service settings of the model
 * @param taskSettings the task settings of the model
 */
public record VoyageAIEmbeddingsRequestEntity(
    List<InferenceStringGroup> input,
    @Nullable InputType inputType,
    BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
    VoyageAIEmbeddingsTaskSettings taskSettings
) implements ToXContentObject {

    // Accepted input_type values
    private static final String INPUT_TYPE_DOCUMENT_VALUE = "document";
    private static final String INPUT_TYPE_QUERY_VALUE = "query";

    // Field names for request body
    public static final String INPUT_FIELD = "input";
    public static final String INPUTS_FIELD = "inputs";
    public static final String CONTENT_FIELD = "content";
    public static final String TYPE_FIELD = "type";
    public static final String TEXT_FIELD = "text";
    public static final String TEXT_TYPE = "text";
    public static final String IMAGE_BASE64_FIELD = "image_base64";
    public static final String ENABLE_AUTO_CHUNKING_FIELD = "enable_auto_chunking";
    public static final String CHUNK_SIZE_FIELD = "chunk_size";
    public static final String MODEL_FIELD = "model";
    public static final String INPUT_TYPE_FIELD = "input_type";
    public static final String TRUNCATION_FIELD = "truncation";
    public static final String OUTPUT_DIMENSION_FIELD = "output_dimension";
    public static final String OUTPUT_DTYPE_FIELD = "output_dtype";

    public VoyageAIEmbeddingsRequestEntity {
        Objects.requireNonNull(input);
        Objects.requireNonNull(taskSettings);
        Objects.requireNonNull(serviceSettings);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        // prefer the root level inputType over task settings input type
        var effectiveInputType = InputType.isSpecified(inputType) ? inputType : taskSettings.inputType();
        var inputTypeValue = InputType.isSpecified(effectiveInputType) ? convertInputTypeToString(effectiveInputType) : null;
        if (serviceSettings.isMultimodal()) {
            writeMultimodalInputs(builder);
        } else if (serviceSettings.isContextual()) {
            // Each string is a document. Voyage AI only chunks documents (input type "document"), and rejects automatic chunking
            // for any other input type, so queries are sent as they are.
            builder.field(INPUTS_FIELD, toStringList(input));
            if (INPUT_TYPE_DOCUMENT_VALUE.equals(inputTypeValue)) {
                builder.field(ENABLE_AUTO_CHUNKING_FIELD, true);
                builder.field(CHUNK_SIZE_FIELD, VoyageAIUtils.CONTEXTUALIZED_CHUNK_SIZE);
            }
        } else {
            builder.field(INPUT_FIELD, toStringList(input));
        }
        builder.field(MODEL_FIELD, serviceSettings.modelId());
        if (inputTypeValue != null) {
            builder.field(INPUT_TYPE_FIELD, inputTypeValue);
        }
        if (taskSettings.truncation() != null) {
            builder.field(TRUNCATION_FIELD, taskSettings.truncation());
        }
        if (serviceSettings.dimensions() != null) {
            builder.field(OUTPUT_DIMENSION_FIELD, serviceSettings.dimensions());
        }
        builder.field(OUTPUT_DTYPE_FIELD, serviceSettings.embeddingType().toRequestString());
        builder.endObject();
        return builder;
    }

    private void writeMultimodalInputs(XContentBuilder builder) throws IOException {
        builder.startArray(INPUTS_FIELD);
        for (var inferenceStringGroup : input) {
            builder.startObject();
            builder.startArray(CONTENT_FIELD);
            for (var inferenceString : inferenceStringGroup.inferenceStrings()) {
                writeContentItem(builder, inferenceString);
            }
            builder.endArray();
            builder.endObject();
        }
        builder.endArray();
    }

    private static void writeContentItem(XContentBuilder builder, InferenceString inferenceString) throws IOException {
        builder.startObject();
        switch (inferenceString.dataType()) {
            case TEXT -> {
                builder.field(TYPE_FIELD, TEXT_TYPE);
                builder.field(TEXT_FIELD, inferenceString.value());
            }
            case IMAGE -> {
                if (inferenceString.dataFormat() != DataFormat.BASE64) {
                    throw unsupportedInput(inferenceString);
                }
                builder.field(TYPE_FIELD, IMAGE_BASE64_FIELD);
                builder.field(IMAGE_BASE64_FIELD, inferenceString.value());
            }
            case AUDIO, VIDEO, PDF, HTML -> throw unsupportedInput(inferenceString);
        }
        builder.endObject();
    }

    private static IllegalArgumentException unsupportedInput(InferenceString inferenceString) {
        return new IllegalArgumentException(
            "Voyage AI multimodal embeddings do not support inputs of type ["
                + inferenceString.dataType()
                + "] and format ["
                + inferenceString.dataFormat()
                + "], only text and base64 image inputs are supported"
        );
    }

    public static String convertInputTypeToString(InputType inputType) {
        return switch (inputType) {
            case null -> null;
            case INGEST, INTERNAL_INGEST -> INPUT_TYPE_DOCUMENT_VALUE;
            case SEARCH, INTERNAL_SEARCH -> INPUT_TYPE_QUERY_VALUE;
            default -> {
                assert false : invalidInputTypeMessage(inputType);
                yield null;
            }
        };
    }
}
