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
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.xpack.core.ml.AbstractBWCWireSerializationTestCase;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettingsTests;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.xpack.inference.Utils.randomSimilarityMeasure;
import static org.elasticsearch.xpack.inference.services.ServiceFields.DIMENSIONS_SET_BY_USER;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MULTIMODAL_MODEL;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService.VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIServiceFields.CONTEXTUAL_MODEL;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;

public class VoyageAIEmbeddingServiceSettingsTests extends AbstractBWCWireSerializationTestCase<VoyageAIEmbeddingServiceSettings> {

    private static final String TEST_MODEL_ID = "test-model-id";
    private static final String INITIAL_TEST_MODEL_ID = "initial-test-model-id";

    private static final int TEST_RATE_LIMIT = 20;
    private static final int INITIAL_TEST_RATE_LIMIT = 30;
    private static final int DEFAULT_RATE_LIMIT = 2_000;

    private static final int TEST_DIMENSIONS = 1024;
    private static final int INITIAL_TEST_DIMENSIONS = 2048;

    private static final int TEST_MAX_INPUT_TOKENS = 512;
    private static final int INITIAL_TEST_MAX_INPUT_TOKENS = 1024;

    private static final SimilarityMeasure TEST_SIMILARITY_MEASURE = SimilarityMeasure.COSINE;
    private static final SimilarityMeasure INITIAL_TEST_SIMILARITY_MEASURE = SimilarityMeasure.DOT_PRODUCT;

    private static final VoyageAIEmbeddingType TEST_EMBEDDING_TYPE = VoyageAIEmbeddingType.INT8;
    private static final VoyageAIEmbeddingType INITIAL_TEST_EMBEDDING_TYPE = VoyageAIEmbeddingType.FLOAT;
    private static final VoyageAIEmbeddingType DEFAULT_EMBEDDING_TYPE = VoyageAIEmbeddingType.FLOAT;

    public static VoyageAIEmbeddingServiceSettings createRandom() {
        return new VoyageAIEmbeddingServiceSettings(
            randomAlphaOfLength(15),
            RateLimitSettingsTests.createRandom(),
            randomFrom(VoyageAIEmbeddingType.values()),
            randomFrom(randomSimilarityMeasure(), null),
            randomFrom(randomIntBetween(128, 256), null),
            randomFrom(randomIntBetween(128, 2048), null),
            randomBoolean(),
            randomBoolean()
        );
    }

    public void testFromMap_Request_AllFields_CreatesSettingsCorrectly() {
        var multimodalModel = randomBoolean();
        var serviceSettings = VoyageAIEmbeddingServiceSettings.fromMap(
            buildServiceSettingsMap(
                TEST_MODEL_ID,
                TEST_EMBEDDING_TYPE.toString(),
                TEST_SIMILARITY_MEASURE.toString(),
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                null,
                TEST_RATE_LIMIT,
                multimodalModel
            ),
            ConfigurationParseContext.REQUEST
        );

        assertThat(
            serviceSettings,
            is(
                new VoyageAIEmbeddingServiceSettings(
                    TEST_MODEL_ID,
                    new RateLimitSettings(TEST_RATE_LIMIT),
                    TEST_EMBEDDING_TYPE,
                    TEST_SIMILARITY_MEASURE,
                    TEST_DIMENSIONS,
                    TEST_MAX_INPUT_TOKENS,
                    true,
                    multimodalModel
                )
            )
        );
    }

    public void testFromMap_Persistent_AllFields_CreatesSettingsCorrectly() {
        var dimensionsSetByUser = randomBoolean();
        var multimodalModel = randomBoolean();
        var serviceSettings = VoyageAIEmbeddingServiceSettings.fromMap(
            buildServiceSettingsMap(
                TEST_MODEL_ID,
                TEST_EMBEDDING_TYPE.toString(),
                TEST_SIMILARITY_MEASURE.toString(),
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                dimensionsSetByUser,
                TEST_RATE_LIMIT,
                multimodalModel
            ),
            ConfigurationParseContext.PERSISTENT
        );

        assertThat(
            serviceSettings,
            is(
                new VoyageAIEmbeddingServiceSettings(
                    TEST_MODEL_ID,
                    new RateLimitSettings(TEST_RATE_LIMIT),
                    TEST_EMBEDDING_TYPE,
                    TEST_SIMILARITY_MEASURE,
                    TEST_DIMENSIONS,
                    TEST_MAX_INPUT_TOKENS,
                    dimensionsSetByUser,
                    multimodalModel
                )
            )
        );
    }

    public void testFromMap_OnlyMandatoryFields_UsesDefaults() {
        var context = randomFrom(ConfigurationParseContext.values());
        var serviceSettings = VoyageAIEmbeddingServiceSettings.fromMap(buildServiceSettingsMap(TEST_MODEL_ID), context);

        assertThat(
            serviceSettings,
            is(
                new VoyageAIEmbeddingServiceSettings(
                    TEST_MODEL_ID,
                    new RateLimitSettings(DEFAULT_RATE_LIMIT),
                    DEFAULT_EMBEDDING_TYPE,
                    null,
                    null,
                    null,
                    false,
                    true
                )
            )
        );
        assertTrue(serviceSettings.isMultimodal());
    }

    public void testFromMap_Request_DimensionsSetByUser_ThrowsValidationError() {
        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(
                buildServiceSettingsMap(TEST_MODEL_ID, null, null, TEST_DIMENSIONS, null, randomBoolean(), null, null),
                ConfigurationParseContext.REQUEST
            )
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is(Strings.format("[service_settings] does not allow the setting [%s]", DIMENSIONS_SET_BY_USER))
        );
    }

    public void testFromMap_ContextualModel_IsNotAccepted() {
        var map = buildServiceSettingsMap(TEST_MODEL_ID);
        map.put(CONTEXTUAL_MODEL, true);

        var serviceSettings = VoyageAIEmbeddingServiceSettings.fromMap(map, ConfigurationParseContext.REQUEST);

        // The setting is not consumed, so the service rejects the request as it has unknown settings left over
        assertThat(map, is(Map.of(CONTEXTUAL_MODEL, true)));
        assertFalse(serviceSettings.isContextual());
    }

    public void testFromMap_MultimodalModel_InvalidType_ThrowsValidationError() {
        var map = buildServiceSettingsMap(TEST_MODEL_ID);
        map.put(MULTIMODAL_MODEL, "yes");

        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(map, randomFrom(ConfigurationParseContext.values()))
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is("field [multimodal_model] is not of the expected type. The value [yes] cannot be converted to a [Boolean]")
        );
    }

    public void testFromMap_InvalidEmbeddingType_ThrowsValidationError() {
        var invalidEmbeddingType = "not_a_type";
        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(
                buildServiceSettingsMap(TEST_MODEL_ID, invalidEmbeddingType, null, null, null, null, null, null),
                randomFrom(ConfigurationParseContext.values())
            )
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is(
                Strings.format(
                    "[service_settings] Invalid value [%s] received. [embedding_type] must be one of [binary, bit, byte, float, int8]",
                    invalidEmbeddingType
                )
            )
        );
    }

    public void testFromMap_InvalidSimilarity_ThrowsValidationError() {
        var invalidSimilarity = "by_size";
        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(
                buildServiceSettingsMap(TEST_MODEL_ID, null, invalidSimilarity, null, null, null, null, null),
                randomFrom(ConfigurationParseContext.values())
            )
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is(
                Strings.format(
                    "[service_settings] Invalid value [%s] received. [similarity] must be one of [cosine, dot_product, l2_norm]",
                    invalidSimilarity
                )
            )
        );
    }

    public void testFromMap_InvalidDimensions_ThrowsValidationError() {
        var invalidDimensions = randomIntBetween(-10, 0);
        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(
                buildServiceSettingsMap(TEST_MODEL_ID, null, null, invalidDimensions, null, null, null, null),
                ConfigurationParseContext.PERSISTENT
            )
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is(Strings.format("[service_settings] Invalid value [%d]. [dimensions] must be a positive integer", invalidDimensions))
        );
    }

    public void testFromMap_InvalidMaxInputTokens_ThrowsValidationError() {
        var invalidMaxInputTokens = randomIntBetween(-10, 0);
        var thrownException = expectThrows(
            ValidationException.class,
            () -> VoyageAIEmbeddingServiceSettings.fromMap(
                buildServiceSettingsMap(TEST_MODEL_ID, null, null, null, invalidMaxInputTokens, null, null, null),
                randomFrom(ConfigurationParseContext.values())
            )
        );

        assertThat(thrownException.validationErrors().size(), is(1));
        assertThat(
            thrownException.validationErrors().getFirst(),
            is(
                Strings.format(
                    "[service_settings] Invalid value [%d]. [max_input_tokens] must be a positive integer",
                    invalidMaxInputTokens
                )
            )
        );
    }

    public void testUpdateServiceSettings_AllFields_OnlyMutableFieldsAreUpdated() {
        var initialDimensionsSetByUser = randomBoolean();
        var initialMultimodalModel = randomBoolean();
        var originalServiceSettings = new VoyageAIEmbeddingServiceSettings(
            INITIAL_TEST_MODEL_ID,
            new RateLimitSettings(INITIAL_TEST_RATE_LIMIT),
            INITIAL_TEST_EMBEDDING_TYPE,
            INITIAL_TEST_SIMILARITY_MEASURE,
            INITIAL_TEST_DIMENSIONS,
            INITIAL_TEST_MAX_INPUT_TOKENS,
            initialDimensionsSetByUser,
            initialMultimodalModel
        );

        var updatedServiceSettings = originalServiceSettings.updateServiceSettings(
            buildServiceSettingsMap(
                TEST_MODEL_ID,
                TEST_EMBEDDING_TYPE.toString(),
                TEST_SIMILARITY_MEASURE.toString(),
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                initialDimensionsSetByUser == false,
                TEST_RATE_LIMIT,
                initialMultimodalModel == false
            )
        );

        assertThat(
            updatedServiceSettings,
            is(
                new VoyageAIEmbeddingServiceSettings(
                    INITIAL_TEST_MODEL_ID,
                    new RateLimitSettings(TEST_RATE_LIMIT),
                    INITIAL_TEST_EMBEDDING_TYPE,
                    INITIAL_TEST_SIMILARITY_MEASURE,
                    INITIAL_TEST_DIMENSIONS,
                    TEST_MAX_INPUT_TOKENS,
                    initialDimensionsSetByUser,
                    initialMultimodalModel
                )
            )
        );
    }

    public void testUpdateServiceSettings_EmptyMap_DoesNotChangeSettings() {
        var originalServiceSettings = createRandom();

        assertThat(originalServiceSettings.updateServiceSettings(new HashMap<>()), is(originalServiceSettings));
    }

    public void testUpdateEmbeddingDetails_SimilarityAndDimensionsAlreadySet_ReturnsExistingSettings() {
        var existingSettings = new VoyageAIEmbeddingServiceSettings(
            TEST_MODEL_ID,
            new RateLimitSettings(TEST_RATE_LIMIT),
            TEST_EMBEDDING_TYPE,
            TEST_SIMILARITY_MEASURE,
            TEST_DIMENSIONS,
            TEST_MAX_INPUT_TOKENS,
            randomBoolean(),
            randomBoolean()
        );

        assertThat(
            BaseVoyageAIEmbeddingsServiceSettings.updateEmbeddingDetails(existingSettings, TEST_DIMENSIONS, TEST_SIMILARITY_MEASURE),
            sameInstance(existingSettings)
        );
    }

    public void testUpdateEmbeddingDetails_DifferentSimilarityAndDimensions_OnlyThoseFieldsAreUpdated() {
        var dimensionsSetByUser = randomBoolean();
        var multimodalModel = randomBoolean();
        var existingSettings = new VoyageAIEmbeddingServiceSettings(
            TEST_MODEL_ID,
            new RateLimitSettings(TEST_RATE_LIMIT),
            TEST_EMBEDDING_TYPE,
            null,
            null,
            TEST_MAX_INPUT_TOKENS,
            dimensionsSetByUser,
            multimodalModel
        );

        var updatedSettings = BaseVoyageAIEmbeddingsServiceSettings.updateEmbeddingDetails(
            existingSettings,
            TEST_DIMENSIONS,
            TEST_SIMILARITY_MEASURE
        );

        assertThat(
            updatedSettings,
            is(
                new VoyageAIEmbeddingServiceSettings(
                    TEST_MODEL_ID,
                    new RateLimitSettings(TEST_RATE_LIMIT),
                    TEST_EMBEDDING_TYPE,
                    TEST_SIMILARITY_MEASURE,
                    TEST_DIMENSIONS,
                    TEST_MAX_INPUT_TOKENS,
                    dimensionsSetByUser,
                    multimodalModel
                )
            )
        );
    }

    public void testToXContent_WritesAllValues() throws IOException {
        var dimensionsSetByUser = randomBoolean();
        var multimodalModel = randomBoolean();
        var serviceSettings = new VoyageAIEmbeddingServiceSettings(
            TEST_MODEL_ID,
            new RateLimitSettings(TEST_RATE_LIMIT),
            TEST_EMBEDDING_TYPE,
            TEST_SIMILARITY_MEASURE,
            TEST_DIMENSIONS,
            TEST_MAX_INPUT_TOKENS,
            dimensionsSetByUser,
            multimodalModel
        );

        assertThat(
            Strings.toString(serviceSettings),
            is(
                XContentHelper.stripWhitespace(
                    Strings.format(
                        """
                            {
                                "model_id": "%s",
                                "rate_limit": {
                                    "requests_per_minute": %d
                                },
                                "similarity": "%s",
                                "dimensions": %d,
                                "max_input_tokens": %d,
                                "embedding_type": "%s",
                                "multimodal_model": %b,
                                "dimensions_set_by_user": %b
                            }
                            """,
                        TEST_MODEL_ID,
                        TEST_RATE_LIMIT,
                        TEST_SIMILARITY_MEASURE,
                        TEST_DIMENSIONS,
                        TEST_MAX_INPUT_TOKENS,
                        TEST_EMBEDDING_TYPE,
                        multimodalModel,
                        dimensionsSetByUser
                    )
                )
            )
        );
    }

    public void testToXContent_OnlyMandatoryFields_WritesOnlyMandatoryFieldsAndDefaults() throws IOException {
        var dimensionsSetByUser = randomBoolean();
        var serviceSettings = new VoyageAIEmbeddingServiceSettings(
            TEST_MODEL_ID,
            null,
            TEST_EMBEDDING_TYPE,
            null,
            null,
            null,
            dimensionsSetByUser,
            VoyageAIEmbeddingServiceSettings.DEFAULT_MULTIMODAL_MODEL
        );

        assertThat(Strings.toString(serviceSettings), is(XContentHelper.stripWhitespace(Strings.format("""
            {
                "model_id": "%s",
                "rate_limit": {
                    "requests_per_minute": %d
                },
                "embedding_type": "%s",
                "multimodal_model": true,
                "dimensions_set_by_user": %b
            }
            """, TEST_MODEL_ID, DEFAULT_RATE_LIMIT, TEST_EMBEDDING_TYPE, dimensionsSetByUser))));
    }

    public void testIsContextual_IsAlwaysFalse() {
        assertFalse(createRandom().isContextual());
    }

    public void testSupportsVersion() {
        var serviceSettings = createRandom();

        assertTrue(serviceSettings.supportsVersion(TransportVersion.current()));
        assertFalse(serviceSettings.supportsVersion(TransportVersion.minimumCompatible()));
    }

    @Override
    protected Writeable.Reader<VoyageAIEmbeddingServiceSettings> instanceReader() {
        return VoyageAIEmbeddingServiceSettings::new;
    }

    @Override
    protected VoyageAIEmbeddingServiceSettings createTestInstance() {
        return createRandom();
    }

    @Override
    protected VoyageAIEmbeddingServiceSettings mutateInstance(VoyageAIEmbeddingServiceSettings instance) throws IOException {
        var modelId = instance.modelId();
        var rateLimitSettings = instance.rateLimitSettings();
        var embeddingType = instance.embeddingType();
        var similarity = instance.similarity();
        var dimensions = instance.dimensions();
        var maxInputTokens = instance.maxInputTokens();
        var dimensionsSetByUser = instance.dimensionsSetByUser();
        var multimodalModel = instance.isMultimodal();
        switch (randomInt(7)) {
            case 0 -> modelId = randomValueOtherThan(modelId, () -> randomAlphaOfLength(15));
            case 1 -> rateLimitSettings = randomValueOtherThan(rateLimitSettings, RateLimitSettingsTests::createRandom);
            case 2 -> embeddingType = randomValueOtherThan(embeddingType, () -> randomFrom(VoyageAIEmbeddingType.values()));
            case 3 -> similarity = randomValueOtherThan(similarity, () -> randomFrom(randomSimilarityMeasure(), null));
            case 4 -> dimensions = randomValueOtherThan(dimensions, () -> randomFrom(randomIntBetween(1, 2048), null));
            case 5 -> maxInputTokens = randomValueOtherThan(maxInputTokens, () -> randomFrom(randomIntBetween(128, 2048), null));
            case 6 -> dimensionsSetByUser = dimensionsSetByUser == false;
            case 7 -> multimodalModel = multimodalModel == false;
            default -> throw new AssertionError("Illegal randomisation branch");
        }

        return new VoyageAIEmbeddingServiceSettings(
            modelId,
            rateLimitSettings,
            embeddingType,
            similarity,
            dimensions,
            maxInputTokens,
            dimensionsSetByUser,
            multimodalModel
        );
    }

    @Override
    protected VoyageAIEmbeddingServiceSettings mutateInstanceForVersion(
        VoyageAIEmbeddingServiceSettings instance,
        TransportVersion version
    ) {
        // Nodes on a version before the multimodal_model setting was added do not know about it, so it is read back as false
        if (version.supports(VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED) == false && instance.isMultimodal()) {
            return new VoyageAIEmbeddingServiceSettings(
                instance.modelId(),
                instance.rateLimitSettings(),
                instance.embeddingType(),
                instance.similarity(),
                instance.dimensions(),
                instance.maxInputTokens(),
                instance.dimensionsSetByUser(),
                false
            );
        }
        return instance;
    }

    private static Map<String, Object> buildServiceSettingsMap(String modelId) {
        return buildServiceSettingsMap(modelId, null, null, null, null, null, null, null);
    }

    private static Map<String, Object> buildServiceSettingsMap(
        @Nullable String modelId,
        @Nullable String embeddingType,
        @Nullable String similarity,
        @Nullable Integer dimensions,
        @Nullable Integer maxInputTokens,
        @Nullable Boolean dimensionsSetByUser,
        @Nullable Integer rateLimit,
        @Nullable Boolean multimodalModel
    ) {
        var map = new HashMap<String, Object>();
        if (modelId != null) {
            map.put(ServiceFields.MODEL_ID, modelId);
        }
        if (embeddingType != null) {
            map.put(ServiceFields.EMBEDDING_TYPE, embeddingType);
        }
        if (similarity != null) {
            map.put(ServiceFields.SIMILARITY, similarity);
        }
        if (dimensions != null) {
            map.put(ServiceFields.DIMENSIONS, dimensions);
        }
        if (maxInputTokens != null) {
            map.put(ServiceFields.MAX_INPUT_TOKENS, maxInputTokens);
        }
        if (dimensionsSetByUser != null) {
            map.put(DIMENSIONS_SET_BY_USER, dimensionsSetByUser);
        }
        if (rateLimit != null) {
            map.put(RateLimitSettings.FIELD_NAME, new HashMap<>(Map.of(RateLimitSettings.REQUESTS_PER_MINUTE_FIELD, rateLimit)));
        }
        if (multimodalModel != null) {
            map.put(MULTIMODAL_MODEL, multimodalModel);
        }
        return map;
    }
}
