/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.embeddings;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.ServiceUtils;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.voyageai.VoyageAIServiceSettings;

import java.io.IOException;
import java.util.EnumSet;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS_SET_BY_USER;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MAX_INPUT_TOKENS;
import static org.elasticsearch.xpack.inference.services.ServiceFields.SIMILARITY;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalBoolean;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalEnum;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractOptionalPositiveInteger;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractRequiredString;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractSimilarity;
import static org.elasticsearch.xpack.inference.services.SettingsScope.SERVICE_SETTINGS;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_INTEGRATION_ADDED;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED;

/**
 * Abstract base for the Voyage AI embeddings service settings. Extends {@link VoyageAIServiceSettings} and adds the fields that are
 * common to every dense embeddings task type (embedding type, similarity, dimensions, max input tokens). The concrete variants differ in
 * the task type they are used for, and in the flags they carry: {@link VoyageAITextEmbeddingServiceSettings} for the
 * {@code text_embedding} task and {@link VoyageAIEmbeddingServiceSettings} (multimodal) for the {@code embedding} task.
 */
public abstract class BaseVoyageAIEmbeddingsServiceSettings extends VoyageAIServiceSettings {

    private static final TransportVersion VOYAGE_AI_EMBEDDING_TYPE_NON_NULLABLE = TransportVersion.fromName(
        "voyage_ai_embedding_type_non_nullable"
    );

    /**
     * The service settings fields shared by all the embeddings task types, as parsed from a map.
     */
    protected record CommonFields(
        String modelId,
        RateLimitSettings rateLimitSettings,
        VoyageAIEmbeddingType embeddingType,
        @Nullable SimilarityMeasure similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        boolean dimensionsSetByUser
    ) {}

    /**
     * Parses the fields that are common to all the embeddings task types, removing them from {@code map}. Any problems are added to
     * {@code validationException}, which the caller must check once it has also parsed its own fields.
     */
    protected static CommonFields parseCommonFields(
        Map<String, Object> map,
        ConfigurationParseContext context,
        ValidationException validationException
    ) {
        var modelId = extractRequiredString(map, ServiceFields.MODEL_ID, SERVICE_SETTINGS, validationException);
        var rateLimitSettings = extractRateLimitSettings(map, context, validationException);

        var embeddingType = Objects.requireNonNullElse(
            extractOptionalEnum(
                map,
                ServiceFields.EMBEDDING_TYPE,
                SERVICE_SETTINGS,
                VoyageAIEmbeddingType::fromString,
                EnumSet.allOf(VoyageAIEmbeddingType.class),
                validationException
            ),
            VoyageAIEmbeddingType.FLOAT
        );

        var similarity = extractSimilarity(map, SERVICE_SETTINGS, validationException);
        var dimensions = extractOptionalPositiveInteger(map, DIMENSIONS, SERVICE_SETTINGS, validationException);
        var maxInputTokens = extractOptionalPositiveInteger(map, MAX_INPUT_TOKENS, SERVICE_SETTINGS, validationException);

        var dimensionsSetByUser = extractOptionalBoolean(map, DIMENSIONS_SET_BY_USER, validationException);

        switch (context) {
            case REQUEST -> {
                if (dimensionsSetByUser != null) {
                    validationException.addValidationError(ServiceUtils.invalidSettingError(DIMENSIONS_SET_BY_USER, SERVICE_SETTINGS));
                }
                dimensionsSetByUser = dimensions != null;
            }
            case PERSISTENT -> {
                if (dimensionsSetByUser == null) {
                    dimensionsSetByUser = false;
                }
            }
        }

        return new CommonFields(modelId, rateLimitSettings, embeddingType, similarity, dimensions, maxInputTokens, dimensionsSetByUser);
    }

    /**
     * Returns the settings to use once the embedding size is known, which is the existing settings if the similarity and dimensions
     * are already the ones provided.
     */
    public static BaseVoyageAIEmbeddingsServiceSettings updateEmbeddingDetails(
        BaseVoyageAIEmbeddingsServiceSettings existingSettings,
        Integer embeddingSize,
        SimilarityMeasure similarityToUse
    ) {
        if (embeddingSize.equals(existingSettings.dimensions()) && similarityToUse.equals(existingSettings.similarity())) {
            return existingSettings;
        }
        return existingSettings.update(similarityToUse, embeddingSize);
    }

    private final VoyageAIEmbeddingType embeddingType;
    private final SimilarityMeasure similarity;
    private final Integer dimensions;
    private final Integer maxInputTokens;
    private final boolean dimensionsSetByUser;
    private final boolean multimodalModel;

    protected BaseVoyageAIEmbeddingsServiceSettings(
        String modelId,
        @Nullable RateLimitSettings rateLimitSettings,
        VoyageAIEmbeddingType embeddingType,
        @Nullable SimilarityMeasure similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        boolean dimensionsSetByUser,
        boolean multimodalModel
    ) {
        super(modelId, rateLimitSettings);
        this.embeddingType = Objects.requireNonNull(embeddingType);
        this.similarity = similarity;
        this.dimensions = dimensions;
        this.maxInputTokens = maxInputTokens;
        this.dimensionsSetByUser = dimensionsSetByUser;
        this.multimodalModel = multimodalModel;
    }

    protected BaseVoyageAIEmbeddingsServiceSettings(StreamInput in) throws IOException {
        super(in);
        this.similarity = in.readOptionalEnum(SimilarityMeasure.class);
        this.dimensions = in.readOptionalVInt();
        this.maxInputTokens = in.readOptionalVInt();
        if (VOYAGE_AI_EMBEDDING_TYPE_NON_NULLABLE.supports(in.getTransportVersion())) {
            this.embeddingType = in.readEnum(VoyageAIEmbeddingType.class);
        } else {
            this.embeddingType = Objects.requireNonNullElse(in.readOptionalEnum(VoyageAIEmbeddingType.class), VoyageAIEmbeddingType.FLOAT);
        }
        this.dimensionsSetByUser = in.readBoolean();
        if (in.getTransportVersion().supports(VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED)) {
            this.multimodalModel = in.readBoolean();
        } else {
            this.multimodalModel = false;
        }
    }

    /**
     * Returns new settings with the updated similarity and dimensions, and all other fields unchanged.
     *
     * @param similarity the new similarity
     * @param dimensions the new dimensions
     * @return new settings of the same type as this one
     */
    public abstract BaseVoyageAIEmbeddingsServiceSettings update(SimilarityMeasure similarity, Integer dimensions);

    /**
     * @return {@code true} if the model is served by the Voyage AI contextualized embeddings API, which can only be the case for the
     * {@code text_embedding} task
     */
    public abstract boolean isContextual();

    @Override
    public SimilarityMeasure similarity() {
        return similarity;
    }

    @Override
    public Integer dimensions() {
        return dimensions;
    }

    @Override
    public Boolean dimensionsSetByUser() {
        return dimensionsSetByUser;
    }

    public Integer maxInputTokens() {
        return maxInputTokens;
    }

    public VoyageAIEmbeddingType embeddingType() {
        return embeddingType;
    }

    @Override
    public DenseVectorFieldMapper.ElementType elementType() {
        return embeddingType.toElementType();
    }

    @Override
    public boolean isMultimodal() {
        return multimodalModel;
    }

    /**
     * The fields of the embeddings service settings that can be changed after the endpoint has been created.
     */
    protected record UpdatedFields(RateLimitSettings rateLimitSettings, Integer maxInputTokens) {}

    /**
     * Extracts the fields that can be updated ({@code max_input_tokens} and {@code rate_limit}) from {@code serviceSettings}, falling
     * back to the current values for those that are not present.
     */
    protected UpdatedFields extractUpdatedFields(Map<String, Object> serviceSettings) {
        var validationException = new ValidationException();

        var extractedMaxInputTokens = extractOptionalPositiveInteger(
            serviceSettings,
            MAX_INPUT_TOKENS,
            SERVICE_SETTINGS,
            validationException
        );

        var extractedRateLimitSettings = RateLimitSettings.of(
            serviceSettings,
            rateLimitSettings(),
            validationException,
            ConfigurationParseContext.REQUEST
        );

        validationException.throwIfValidationErrorsExist();

        return new UpdatedFields(extractedRateLimitSettings, extractedMaxInputTokens != null ? extractedMaxInputTokens : maxInputTokens);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        toXContentFragmentOfExposedFields(builder, params);
        builder.field(DIMENSIONS_SET_BY_USER, dimensionsSetByUser);
        builder.endObject();
        return builder;
    }

    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        super.toXContentFragmentOfExposedFields(builder, params);
        if (similarity != null) {
            builder.field(SIMILARITY, similarity);
        }
        if (dimensions != null) {
            builder.field(DIMENSIONS, dimensions);
        }
        if (maxInputTokens != null) {
            builder.field(MAX_INPUT_TOKENS, maxInputTokens);
        }
        builder.field(ServiceFields.EMBEDDING_TYPE, embeddingType);
        return builder;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        assert false : "should never be called when supportsVersion is used";
        return VOYAGE_AI_INTEGRATION_ADDED;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalEnum(similarity);
        out.writeOptionalVInt(dimensions);
        out.writeOptionalVInt(maxInputTokens);
        if (VOYAGE_AI_EMBEDDING_TYPE_NON_NULLABLE.supports(out.getTransportVersion())) {
            out.writeEnum(embeddingType);
        } else {
            out.writeOptionalEnum(embeddingType);
        }
        out.writeBoolean(dimensionsSetByUser);
        if (out.getTransportVersion().supports(VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED)) {
            out.writeBoolean(multimodalModel);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BaseVoyageAIEmbeddingsServiceSettings that = (BaseVoyageAIEmbeddingsServiceSettings) o;
        return Objects.equals(modelId(), that.modelId())
            && Objects.equals(rateLimitSettings(), that.rateLimitSettings())
            && Objects.equals(similarity, that.similarity)
            && Objects.equals(dimensions, that.dimensions)
            && Objects.equals(maxInputTokens, that.maxInputTokens)
            && embeddingType == that.embeddingType
            && dimensionsSetByUser == that.dimensionsSetByUser
            && multimodalModel == that.multimodalModel;
    }

    @Override
    public int hashCode() {
        return Objects.hash(
            modelId(),
            rateLimitSettings(),
            similarity,
            dimensions,
            maxInputTokens,
            embeddingType,
            dimensionsSetByUser,
            multimodalModel
        );
    }

    @Override
    public String toString() {
        return Strings.toString(this);
    }
}
