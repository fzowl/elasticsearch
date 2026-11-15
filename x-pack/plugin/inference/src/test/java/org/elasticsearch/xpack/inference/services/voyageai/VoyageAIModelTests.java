/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai;

import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelTests;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.voyageai.rerank.VoyageAIRerankModelTests;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;

public class VoyageAIModelTests extends ESTestCase {

    public void testRateLimitGroupingHash_IsNullSafe_WhenModelIsConstructedWithApiKey() {
        var apiKey = "voyage-api-key";
        var otherApiKey = "different-voyage-api-key";

        // Known model id -> mapped to "embed_medium" via MODEL_TO_MODEL_FAMILY.
        var knownModel = VoyageAIEmbeddingsModelTests.createModel("url", apiKey, null, "voyage-3.5");
        assertThat(knownModel.getSecretSettings(), notNullValue());
        assertThat(knownModel.getSecretSettings().apiKey(), notNullValue());
        var knownHash = knownModel.rateLimitGroupingHash();
        assertThat("hash must be deterministic", knownModel.rateLimitGroupingHash(), equalTo(knownHash));

        // Unknown model id -> falls back to DEFAULT_MODEL_FAMILY without throwing.
        var unknownModel = VoyageAIEmbeddingsModelTests.createModel("url", apiKey, null, "no-such-model");
        assertThat(unknownModel.getSecretSettings(), notNullValue());
        assertThat(unknownModel.getSecretSettings().apiKey(), notNullValue());
        assertThat(unknownModel.rateLimitGroupingHash(), equalTo(unknownModel.rateLimitGroupingHash()));

        // Same family ("embed_medium") + same api key => same group.
        var sameFamilyModel = VoyageAIEmbeddingsModelTests.createModel("url", apiKey, null, "voyage-3");
        assertThat(sameFamilyModel.rateLimitGroupingHash(), equalTo(knownHash));

        // Different api key with the same model => different group.
        var differentKeyModel = VoyageAIEmbeddingsModelTests.createModel("url", otherApiKey, null, "voyage-3.5");
        assertThat(differentKeyModel.rateLimitGroupingHash(), not(equalTo(knownHash)));

        // Rerank concrete subclass also goes through the same VoyageAIModel#rateLimitGroupingHash.
        var rerankModel = VoyageAIRerankModelTests.createModel(apiKey, "rerank-2", null);
        assertThat(rerankModel.getSecretSettings(), notNullValue());
        assertThat(rerankModel.getSecretSettings().apiKey(), notNullValue());
        assertThat(rerankModel.rateLimitGroupingHash(), equalTo(rerankModel.rateLimitGroupingHash()));
    }

    public void testRateLimitGroupingHash_MultimodalModelsShareAFamily() {
        var apiKey = "voyage-api-key";

        var multimodal3 = multimodalModel(apiKey, "voyage-multimodal-3");
        var multimodal35 = multimodalModel(apiKey, "voyage-multimodal-3.5");
        assertThat(multimodal35.rateLimitGroupingHash(), equalTo(multimodal3.rateLimitGroupingHash()));

        // The multimodal models have their own limits, which are not shared with the text models
        var textModel = VoyageAIEmbeddingsModelTests.createModel("url", apiKey, null, "voyage-3");
        assertThat(multimodal35.rateLimitGroupingHash(), not(equalTo(textModel.rateLimitGroupingHash())));
    }

    public void testUri_ForMultimodalModel_IsTheModelUri() {
        var model = multimodalModel("voyage-api-key", "voyage-multimodal-3.5");

        assertThat(model.uri().toString(), equalTo("url"));
    }

    private static VoyageAIEmbeddingsModel multimodalModel(String apiKey, String modelId) {
        return VoyageAIEmbeddingsModelTests.createMultimodalModel(
            "url",
            apiKey,
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            null,
            null,
            modelId
        );
    }
}
