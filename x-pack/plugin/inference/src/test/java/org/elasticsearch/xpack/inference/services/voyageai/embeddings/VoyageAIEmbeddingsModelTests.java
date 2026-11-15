/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.embeddings;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.hamcrest.MatcherAssert;

import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettingsTests.getTaskSettingsMap;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;

public class VoyageAIEmbeddingsModelTests extends ESTestCase {

    public void testOverrideWith_DoesNotOverrideAndModelRemainsEqual_WhenSettingsAreEmpty() {
        var model = createModel("url", "api_key", null, null, "model");

        var overriddenModel = VoyageAIEmbeddingsModel.of(model, Map.of());
        MatcherAssert.assertThat(overriddenModel, is(model));
    }

    public void testOverrideWith_DoesNotOverrideAndModelRemainsEqual_WhenSettingsAreNull() {
        var model = createModel("url", "api_key", null, null, "model");

        var overriddenModel = VoyageAIEmbeddingsModel.of(model, null);
        MatcherAssert.assertThat(overriddenModel, is(model));
    }

    public void testOverrideWith_SetsInputType_FromRequestTaskSettings_IfValid_OverridingStoredTaskSettings() {
        var model = createModel("url", "api_key", new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null), null, null, "model");

        var overriddenModel = VoyageAIEmbeddingsModel.of(model, getTaskSettingsMap(InputType.SEARCH));
        var expectedModel = createModel("url", "api_key", new VoyageAIEmbeddingsTaskSettings(InputType.SEARCH, null), null, null, "model");
        MatcherAssert.assertThat(overriddenModel, is(expectedModel));
    }

    public void testOverrideWith_DoesNotOverrideInputType_WhenRequestTaskSettingsIsNull() {
        var model = createModel("url", "api_key", new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null), null, null, "model");

        var overriddenModel = VoyageAIEmbeddingsModel.of(model, getTaskSettingsMap(null));
        var expectedModel = createModel("url", "api_key", new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null), null, null, "model");
        MatcherAssert.assertThat(overriddenModel, is(expectedModel));
    }

    public void testUri_UsesVoyageHost_ForVoyageApiKey() {
        assertThat(createModelFromMaps("pa-key").uri().toString(), is("https://api.voyageai.com/v1/embeddings"));
    }

    public void testUri_UsesMongoDbHost_ForMongoDbApiKey() {
        assertThat(createModelFromMaps("al-key").uri().toString(), is("https://ai.mongodb.com/v1/embeddings"));
    }

    public void testUri_UsesContextualizedEndpoint_WhenContextualModelIsSet() {
        assertThat(
            createModelFromMaps("pa-key", "voyage-context-4", TaskType.TEXT_EMBEDDING, Map.of("contextual_model", true)).uri().toString(),
            is("https://api.voyageai.com/v1/contextualizedembeddings")
        );
        assertThat(
            createModelFromMaps("al-key", "my-model", TaskType.TEXT_EMBEDDING, Map.of("contextual_model", true)).uri().toString(),
            is("https://ai.mongodb.com/v1/contextualizedembeddings")
        );
    }

    public void testUri_DoesNotUseContextualizedEndpoint_ForContextualModelIdWithoutTheSetting() {
        assertThat(
            createModelFromMaps("pa-key", "voyage-context-4", TaskType.TEXT_EMBEDDING, Map.of()).uri().toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
        assertThat(
            createModelFromMaps("pa-key", "voyage-context-3", TaskType.TEXT_EMBEDDING, Map.of("contextual_model", false)).uri().toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
    }

    public void testUri_UsesMultimodalEndpoint_ForEmbeddingTask() {
        assertThat(
            createModelFromMaps("pa-key", "voyage-multimodal-3.5", TaskType.EMBEDDING, Map.of()).uri().toString(),
            is("https://api.voyageai.com/v1/multimodalembeddings")
        );
        assertThat(
            createModelFromMaps("al-key", "voyage-multimodal-3", TaskType.EMBEDDING, Map.of()).uri().toString(),
            is("https://ai.mongodb.com/v1/multimodalembeddings")
        );
    }

    public void testUri_UsesEmbeddingsEndpoint_ForEmbeddingTask_WhenMultimodalModelIsFalse() {
        assertThat(
            createModelFromMaps("pa-key", "voyage-4", TaskType.EMBEDDING, Map.of("multimodal_model", false)).uri().toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
    }

    public void testUri_DoesNotUseMultimodalEndpoint_ForTextEmbeddingTask() {
        assertThat(
            createModelFromMaps("pa-key", "voyage-multimodal-3", TaskType.TEXT_EMBEDDING, Map.of()).uri().toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
    }

    public void testUri_IsBuiltFromTheSettings_WhenModelIsCreatedFromConfigurationsAndSecrets() {
        var secrets = new ModelSecrets(new DefaultSecretSettings(new SecureString("al-key".toCharArray())));

        var multimodalModel = createMultimodalModel("url", "pa-key", VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS, null, null, "model");
        var multimodalConfigurations = multimodalModel.getConfigurations();
        assertThat(
            new VoyageAIEmbeddingsModel(multimodalConfigurations, secrets).uri().toString(),
            is("https://ai.mongodb.com/v1/multimodalembeddings")
        );

        var contextualModel = createContextualModel("url", "pa-key", "model");
        assertThat(
            new VoyageAIEmbeddingsModel(contextualModel.getConfigurations(), secrets).uri().toString(),
            is("https://ai.mongodb.com/v1/contextualizedembeddings")
        );

        var textModel = createModel("url", "pa-key", null, "model");
        assertThat(
            new VoyageAIEmbeddingsModel(textModel.getConfigurations(), secrets).uri().toString(),
            is("https://ai.mongodb.com/v1/embeddings")
        );
    }

    public void testCreateModel_ForEmbeddingTask_UsesMultimodalServiceSettings() {
        var model = createModelFromMaps("pa-key", "voyage-multimodal-3.5", TaskType.EMBEDDING, Map.of());

        assertThat(model.getTaskType(), is(TaskType.EMBEDDING));
        assertThat(model.getServiceSettings(), instanceOf(VoyageAIEmbeddingServiceSettings.class));
        assertTrue(model.getServiceSettings().isMultimodal());
    }

    public void testCreateModel_ForTextEmbeddingTask_UsesTextEmbeddingServiceSettings() {
        var model = createModelFromMaps("pa-key", "voyage-4", TaskType.TEXT_EMBEDDING, Map.of());

        assertThat(model.getTaskType(), is(TaskType.TEXT_EMBEDDING));
        assertThat(model.getServiceSettings(), instanceOf(VoyageAITextEmbeddingServiceSettings.class));
        assertFalse(model.getServiceSettings().isMultimodal());
    }

    public void testCreateModel_ForUnsupportedTaskType_Throws() {
        var exception = expectThrows(
            IllegalArgumentException.class,
            () -> createModelFromMaps("pa-key", "rerank-2", TaskType.RERANK, Map.of())
        );

        assertThat(exception.getMessage(), is("Unsupported task type for Voyage AI embeddings: rerank"));
    }

    public void testOverrideWith_ReturnsModelOfTheSameTaskTypeAndServiceSettings() {
        var model = createMultimodalModel("url", "api_key", VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS, null, null, "model");

        var overriddenModel = VoyageAIEmbeddingsModel.of(model, getTaskSettingsMap(InputType.SEARCH));

        assertThat(overriddenModel.getTaskType(), is(TaskType.EMBEDDING));
        assertThat(overriddenModel.getServiceSettings(), is(model.getServiceSettings()));
        assertThat(overriddenModel.uri(), is(model.uri()));
    }

    private static VoyageAIEmbeddingsModel createModelFromMaps(String apiKey) {
        return createModelFromMaps(apiKey, "voyage-4", TaskType.TEXT_EMBEDDING, Map.of());
    }

    private static VoyageAIEmbeddingsModel createModelFromMaps(
        String apiKey,
        String modelId,
        TaskType taskType,
        Map<String, Object> additionalServiceSettings
    ) {
        var serviceSettings = new HashMap<String, Object>(additionalServiceSettings);
        serviceSettings.put("model_id", modelId);
        return new VoyageAIEmbeddingsModel(
            "id",
            "voyageai",
            serviceSettings,
            new HashMap<String, Object>(),
            null,
            new HashMap<String, Object>(Map.of("api_key", apiKey)),
            ConfigurationParseContext.REQUEST,
            taskType
        );
    }

    public static VoyageAIEmbeddingsModel createModel(String url, String apiKey, @Nullable Integer tokenLimit, @Nullable String model) {
        return createModel(url, apiKey, VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS, tokenLimit, null, model);
    }

    public static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        @Nullable Integer tokenLimit,
        @Nullable Integer dimensions,
        String model
    ) {
        return createModel(url, apiKey, VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS, tokenLimit, dimensions, model);
    }

    public static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable Integer tokenLimit,
        @Nullable Integer dimensions,
        String model
    ) {
        return createModel(
            url,
            apiKey,
            new VoyageAITextEmbeddingServiceSettings(
                model,
                null,
                VoyageAIEmbeddingType.FLOAT,
                SimilarityMeasure.DOT_PRODUCT,
                dimensions,
                tokenLimit,
                false
            ),
            taskSettings,
            chunkingSettings,
            TaskType.TEXT_EMBEDDING
        );
    }

    public static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        @Nullable Integer tokenLimit,
        @Nullable Integer dimensions,
        String model
    ) {
        return createModel(url, apiKey, taskSettings, null, tokenLimit, dimensions, model);
    }

    public static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        @Nullable Integer tokenLimit,
        @Nullable Integer dimensions,
        String model,
        VoyageAIEmbeddingType embeddingType
    ) {
        return createModel(
            url,
            apiKey,
            new VoyageAITextEmbeddingServiceSettings(
                model,
                null,
                embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                dimensions,
                tokenLimit,
                false
            ),
            taskSettings,
            null,
            TaskType.TEXT_EMBEDDING
        );
    }

    public static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        @Nullable Integer tokenLimit,
        @Nullable Integer dimensions,
        String model,
        @Nullable SimilarityMeasure similarityMeasure
    ) {
        return createModel(
            url,
            apiKey,
            new VoyageAITextEmbeddingServiceSettings(
                model,
                null,
                VoyageAIEmbeddingType.FLOAT,
                similarityMeasure,
                dimensions,
                tokenLimit,
                false
            ),
            taskSettings,
            null,
            TaskType.TEXT_EMBEDDING
        );
    }

    /**
     * Creates a model for the {@code text_embedding} task that has the {@code contextual_model} setting, and so is served by the
     * contextualized embeddings API.
     */
    public static VoyageAIEmbeddingsModel createContextualModel(String url, String apiKey, String model) {
        return createModel(
            url,
            apiKey,
            new VoyageAITextEmbeddingServiceSettings(
                model,
                null,
                VoyageAIEmbeddingType.FLOAT,
                SimilarityMeasure.DOT_PRODUCT,
                null,
                null,
                false,
                true
            ),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            null,
            TaskType.TEXT_EMBEDDING
        );
    }

    /**
     * Creates a model for the {@code embedding} task, with the {@code multimodal_model} setting, so it is served by the multimodal
     * embeddings API.
     */
    public static VoyageAIEmbeddingsModel createMultimodalModel(
        String url,
        String apiKey,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        @Nullable Integer dimensions,
        @Nullable VoyageAIEmbeddingType embeddingType,
        String model
    ) {
        return createModel(
            url,
            apiKey,
            new VoyageAIEmbeddingServiceSettings(
                model,
                null,
                embeddingType == null ? VoyageAIEmbeddingType.FLOAT : embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                dimensions,
                null,
                dimensions != null,
                true
            ),
            taskSettings,
            null,
            TaskType.EMBEDDING
        );
    }

    private static VoyageAIEmbeddingsModel createModel(
        String url,
        String apiKey,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        @Nullable ChunkingSettings chunkingSettings,
        TaskType taskType
    ) {
        return new VoyageAIEmbeddingsModel(
            "id",
            "service",
            url,
            serviceSettings,
            taskSettings,
            chunkingSettings,
            new DefaultSecretSettings(new SecureString(apiKey.toCharArray())),
            taskType
        );
    }
}
