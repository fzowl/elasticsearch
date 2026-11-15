/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.embeddings;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;

import java.io.IOException;
import java.util.Map;

import static org.elasticsearch.xpack.inference.services.ServiceFields.MULTIMODAL_MODEL;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalBoolean;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED;

/**
 * Service settings for the {@code embedding} task type, which supports inputs that mix text and images. The {@code multimodal_model} flag
 * defaults to {@code true} and decides the endpoint and the request body format; it is never inferred from the model id. These settings
 * do not accept {@code contextual_model}: a contextual model cannot be multimodal.
 */
public class VoyageAIEmbeddingServiceSettings extends BaseVoyageAIEmbeddingsServiceSettings {
    public static final String NAME = "voyageai_multimodal_embedding_service_settings";
    public static final boolean DEFAULT_MULTIMODAL_MODEL = true;

    public static VoyageAIEmbeddingServiceSettings fromMap(Map<String, Object> map, ConfigurationParseContext context) {
        var validationException = new ValidationException();

        var commonFields = parseCommonFields(map, context, validationException);
        var multimodalModel = extractOptionalBoolean(map, MULTIMODAL_MODEL, validationException);

        validationException.throwIfValidationErrorsExist();

        return new VoyageAIEmbeddingServiceSettings(
            commonFields.modelId(),
            commonFields.rateLimitSettings(),
            commonFields.embeddingType(),
            commonFields.similarity(),
            commonFields.dimensions(),
            commonFields.maxInputTokens(),
            commonFields.dimensionsSetByUser(),
            multimodalModel != null ? multimodalModel : DEFAULT_MULTIMODAL_MODEL
        );
    }

    public VoyageAIEmbeddingServiceSettings(
        String modelId,
        @Nullable RateLimitSettings rateLimitSettings,
        VoyageAIEmbeddingType embeddingType,
        @Nullable SimilarityMeasure similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        boolean dimensionsSetByUser,
        boolean multimodalModel
    ) {
        super(modelId, rateLimitSettings, embeddingType, similarity, dimensions, maxInputTokens, dimensionsSetByUser, multimodalModel);
    }

    public VoyageAIEmbeddingServiceSettings(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    public boolean isContextual() {
        return false;
    }

    @Override
    public BaseVoyageAIEmbeddingsServiceSettings update(SimilarityMeasure similarity, Integer dimensions) {
        return new VoyageAIEmbeddingServiceSettings(
            modelId(),
            rateLimitSettings(),
            embeddingType(),
            similarity,
            dimensions,
            maxInputTokens(),
            dimensionsSetByUser(),
            isMultimodal()
        );
    }

    @Override
    public VoyageAIEmbeddingServiceSettings updateServiceSettings(Map<String, Object> serviceSettings) {
        var updatedFields = extractUpdatedFields(serviceSettings);

        return new VoyageAIEmbeddingServiceSettings(
            modelId(),
            updatedFields.rateLimitSettings(),
            embeddingType(),
            similarity(),
            dimensions(),
            updatedFields.maxInputTokens(),
            dimensionsSetByUser(),
            isMultimodal()
        );
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        super.toXContentFragmentOfExposedFields(builder, params);
        builder.field(MULTIMODAL_MODEL, isMultimodal());
        return builder;
    }

    @Override
    public boolean supportsVersion(TransportVersion version) {
        return version.supports(VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED);
    }
}
