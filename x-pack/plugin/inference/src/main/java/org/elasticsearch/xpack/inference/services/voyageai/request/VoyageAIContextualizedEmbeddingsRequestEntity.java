/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.elasticsearch.inference.InputType;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIEmbeddingsRequestEntity.convertInputTypeToString;

/**
 * Serializes a request body for VoyageAI's contextualized chunk embeddings API
 * (<a href="https://docs.voyageai.com/docs/contextualized-chunk-embeddings">docs</a>).
 * <p>
 * Per the official specification, the {@code inputs} parameter accepts
 * {@code Union[List[List[str]], List[str]]}: either a list of documents where each document is a list of its chunks
 * ({@code List[List[str]]}), or a flat list of chunks belonging to a single document ({@code List[str]}). This entity
 * stores the fully-qualified {@code List[List[str]]} form and can serialize either wire shape, so both supported input
 * formats round-trip correctly. Use {@link #fromSingleDocument} when the caller only has a flat list of chunks.
 */
public record VoyageAIContextualizedEmbeddingsRequestEntity(
    List<List<String>> inputs,
    boolean flattenSingleDocument,
    InputType inputType,
    VoyageAIEmbeddingsServiceSettings serviceSettings,
    VoyageAIEmbeddingsTaskSettings taskSettings
) implements ToXContentObject {

    // Field names for request body
    public static final String INPUTS_FIELD = "inputs";
    public static final String MODEL_FIELD = "model";
    public static final String INPUT_TYPE_FIELD = "input_type";
    public static final String TRUNCATION_FIELD = "truncation";
    public static final String OUTPUT_DIMENSION_FIELD = "output_dimension";
    public static final String OUTPUT_DTYPE_FIELD = "output_dtype";

    public VoyageAIContextualizedEmbeddingsRequestEntity {
        Objects.requireNonNull(inputs);
        Objects.requireNonNull(taskSettings);
        Objects.requireNonNull(serviceSettings);
    }

    /**
     * Builds an entity from a flat list of chunks belonging to a single document. The chunks are serialized either as a
     * flat {@code List[str]} ({@code flatten == true}) or as a single-document {@code List[List[str]]}
     * ({@code flatten == false}); both are accepted by the API.
     */
    public static VoyageAIContextualizedEmbeddingsRequestEntity fromSingleDocument(
        List<String> chunks,
        boolean flatten,
        InputType inputType,
        VoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings
    ) {
        return new VoyageAIContextualizedEmbeddingsRequestEntity(
            List.of(chunks),
            flatten,
            inputType,
            serviceSettings,
            taskSettings
        );
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (flattenSingleDocument && inputs.size() == 1) {
            // List[str] form: a flat list of chunks for a single document
            builder.field(INPUTS_FIELD, inputs.get(0));
        } else {
            // List[List[str]] form: a list of documents, each a list of chunks
            builder.field(INPUTS_FIELD, inputs);
        }
        builder.field(MODEL_FIELD, serviceSettings.modelId());
        // prefer the root level inputType over task settings input type
        if (InputType.isSpecified(inputType)) {
            builder.field(INPUT_TYPE_FIELD, convertInputTypeToString(inputType));
        } else if (InputType.isSpecified(taskSettings.inputType())) {
            builder.field(INPUT_TYPE_FIELD, convertInputTypeToString(taskSettings.inputType()));
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
}
