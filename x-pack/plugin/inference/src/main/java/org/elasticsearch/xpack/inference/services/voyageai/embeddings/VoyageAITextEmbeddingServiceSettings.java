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
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;

import java.io.IOException;
import java.util.Map;

import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalBoolean;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_CONTEXTUAL_EMBEDDINGS_ADDED;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_INTEGRATION_ADDED;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIServiceFields.CONTEXTUAL_MODEL;

/**
 * Service settings for the {@code text_embedding} task type. In addition to the common embeddings settings, they carry the optional
 * {@code contextual_model} flag, which selects the Voyage AI contextualized embeddings API. The flag is never inferred from the model id.
 */
public class VoyageAITextEmbeddingServiceSettings extends BaseVoyageAIEmbeddingsServiceSettings {
    /**
     * This name is a holdover from before the introduction of {@link VoyageAIEmbeddingServiceSettings} to support multimodal embeddings.
     * This name cannot be changed due to backwards compatibility, but it should be 'voyageai_text_embedding_service_settings'.
     */
    public static final String NAME = "voyageai_embeddings_service_settings";

    /** A text embedding model is never multimodal. */
    public static final boolean DEFAULT_MULTIMODAL_MODEL = false;
    public static final boolean DEFAULT_CONTEXTUAL_MODEL = false;

    public static VoyageAITextEmbeddingServiceSettings fromMap(Map<String, Object> map, ConfigurationParseContext context) {
        var validationException = new ValidationException();

        var commonFields = parseCommonFields(map, context, validationException);
        var contextualModel = extractOptionalBoolean(map, CONTEXTUAL_MODEL, validationException);

        validationException.throwIfValidationErrorsExist();

        return new VoyageAITextEmbeddingServiceSettings(
            commonFields.modelId(),
            commonFields.rateLimitSettings(),
            commonFields.embeddingType(),
            commonFields.similarity(),
            commonFields.dimensions(),
            commonFields.maxInputTokens(),
            commonFields.dimensionsSetByUser(),
            contextualModel != null ? contextualModel : DEFAULT_CONTEXTUAL_MODEL
        );
    }

    private final boolean contextualModel;

    public VoyageAITextEmbeddingServiceSettings(
        String modelId,
        @Nullable RateLimitSettings rateLimitSettings,
        VoyageAIEmbeddingType embeddingType,
        @Nullable SimilarityMeasure similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        boolean dimensionsSetByUser
    ) {
        this(
            modelId,
            rateLimitSettings,
            embeddingType,
            similarity,
            dimensions,
            maxInputTokens,
            dimensionsSetByUser,
            DEFAULT_CONTEXTUAL_MODEL
        );
    }

    public VoyageAITextEmbeddingServiceSettings(
        String modelId,
        @Nullable RateLimitSettings rateLimitSettings,
        VoyageAIEmbeddingType embeddingType,
        @Nullable SimilarityMeasure similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        boolean dimensionsSetByUser,
        boolean contextualModel
    ) {
        super(
            modelId,
            rateLimitSettings,
            embeddingType,
            similarity,
            dimensions,
            maxInputTokens,
            dimensionsSetByUser,
            DEFAULT_MULTIMODAL_MODEL
        );
        this.contextualModel = contextualModel;
    }

    public VoyageAITextEmbeddingServiceSettings(StreamInput in) throws IOException {
        super(in);
        if (in.getTransportVersion().supports(VOYAGE_AI_CONTEXTUAL_EMBEDDINGS_ADDED)) {
            this.contextualModel = in.readBoolean();
        } else {
            this.contextualModel = DEFAULT_CONTEXTUAL_MODEL;
        }
    }

    @Override
    public boolean isContextual() {
        return contextualModel;
    }

    @Override
    public BaseVoyageAIEmbeddingsServiceSettings update(SimilarityMeasure similarity, Integer dimensions) {
        return new VoyageAITextEmbeddingServiceSettings(
            modelId(),
            rateLimitSettings(),
            embeddingType(),
            similarity,
            dimensions,
            maxInputTokens(),
            dimensionsSetByUser(),
            contextualModel
        );
    }

    @Override
    public VoyageAITextEmbeddingServiceSettings updateServiceSettings(Map<String, Object> serviceSettings) {
        var updatedFields = extractUpdatedFields(serviceSettings);

        return new VoyageAITextEmbeddingServiceSettings(
            modelId(),
            updatedFields.rateLimitSettings(),
            embeddingType(),
            similarity(),
            dimensions(),
            updatedFields.maxInputTokens(),
            dimensionsSetByUser(),
            contextualModel
        );
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        super.toXContentFragmentOfExposedFields(builder, params);
        // Only write the flag when it is set, so that the settings of existing endpoints, which never had it, are unchanged.
        if (contextualModel) {
            builder.field(CONTEXTUAL_MODEL, true);
        }
        return builder;
    }

    @Override
    public boolean supportsVersion(TransportVersion version) {
        return version.supports(VOYAGE_AI_INTEGRATION_ADDED);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        if (out.getTransportVersion().supports(VOYAGE_AI_CONTEXTUAL_EMBEDDINGS_ADDED)) {
            out.writeBoolean(contextualModel);
        }
    }

    @Override
    public boolean equals(Object o) {
        return super.equals(o) && contextualModel == ((VoyageAITextEmbeddingServiceSettings) o).contextualModel;
    }

    @Override
    public int hashCode() {
        return 31 * super.hashCode() + Boolean.hashCode(contextualModel);
    }
}
