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
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.hamcrest.MatcherAssert;

import java.util.Map;

import static org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettingsTests.getTaskSettingsMap;
import static org.hamcrest.Matchers.is;

public class VoyageAIEmbeddingsModelTests extends ESTestCase {

    public void testContextualizedEmbeddingsModel_RoutesToContextualizedEndpoint() throws Exception {
        MatcherAssert.assertThat(VoyageAIEmbeddingsModel.isContextualizedEmbeddingsModel("voyage-context-3"), is(true));
        MatcherAssert.assertThat(VoyageAIEmbeddingsModel.isContextualizedEmbeddingsModel("voyage-context-4"), is(true));
        MatcherAssert.assertThat(VoyageAIEmbeddingsModel.isContextualizedEmbeddingsModel("voyage-3-large"), is(false));
        MatcherAssert.assertThat(VoyageAIEmbeddingsModel.isContextualizedEmbeddingsModel(null), is(false));

        var voyageKey = new SecureString("regular-key".toCharArray());
        MatcherAssert.assertThat(
            VoyageAIEmbeddingsModel.buildRequestUri("voyage-context-3", voyageKey).toString(),
            is("https://api.voyageai.com/v1/contextualizedembeddings")
        );
        MatcherAssert.assertThat(
            VoyageAIEmbeddingsModel.buildRequestUri("voyage-3-large", voyageKey).toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
        MatcherAssert.assertThat(
            VoyageAIEmbeddingsModel.buildRequestUri("voyage-3-large", null).toString(),
            is("https://api.voyageai.com/v1/embeddings")
        );
    }

    public void testBuildRequestUri_RoutesMongoDBApiKeyToMongoDBHost() throws Exception {
        // MongoDB-issued keys are prefixed with "al-" and must be routed to the MongoDB endpoint, mirroring the
        // official voyageai-python client's get_default_base_url().
        var mongoKey = new SecureString("al-secret-key".toCharArray());
        MatcherAssert.assertThat(
            VoyageAIEmbeddingsModel.buildRequestUri("voyage-3-large", mongoKey).toString(),
            is("https://ai.mongodb.com/v1/embeddings")
        );
        MatcherAssert.assertThat(
            VoyageAIEmbeddingsModel.buildRequestUri("voyage-context-3", mongoKey).toString(),
            is("https://ai.mongodb.com/v1/contextualizedembeddings")
        );
    }

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
        return new VoyageAIEmbeddingsModel(
            "id",
            "service",
            url,
            new VoyageAIEmbeddingsServiceSettings(
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
            new DefaultSecretSettings(new SecureString(apiKey.toCharArray()))
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
        return new VoyageAIEmbeddingsModel(
            "id",
            "service",
            url,
            new VoyageAIEmbeddingsServiceSettings(
                model,
                null,
                VoyageAIEmbeddingType.FLOAT,
                SimilarityMeasure.DOT_PRODUCT,
                dimensions,
                tokenLimit,
                false
            ),
            taskSettings,
            null,
            new DefaultSecretSettings(new SecureString(apiKey.toCharArray()))
        );
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
        return new VoyageAIEmbeddingsModel(
            "id",
            "service",
            url,
            new VoyageAIEmbeddingsServiceSettings(model, null, embeddingType, SimilarityMeasure.DOT_PRODUCT, dimensions, tokenLimit, false),
            taskSettings,
            null,
            new DefaultSecretSettings(new SecureString(apiKey.toCharArray()))
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
        return new VoyageAIEmbeddingsModel(
            "id",
            "service",
            url,
            new VoyageAIEmbeddingsServiceSettings(
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
            new DefaultSecretSettings(new SecureString(apiKey.toCharArray()))
        );
    }
}
