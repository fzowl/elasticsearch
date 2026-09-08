/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.embeddings;

import org.apache.http.client.utils.URIBuilder;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.xpack.inference.external.action.ExecutableAction;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.ServiceUtils;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.elasticsearch.xpack.inference.services.voyageai.VoyageAIModel;
import org.elasticsearch.xpack.inference.services.voyageai.VoyageAIService;
import org.elasticsearch.xpack.inference.services.voyageai.action.VoyageAIActionVisitor;
import org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIUtils;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map;

import static org.elasticsearch.xpack.inference.external.request.RequestUtils.buildUri;
import static org.elasticsearch.xpack.inference.services.voyageai.request.VoyageAIUtils.HOST;

public class VoyageAIEmbeddingsModel extends VoyageAIModel {
    public static VoyageAIEmbeddingsModel of(VoyageAIEmbeddingsModel model, Map<String, Object> taskSettings) {
        var requestTaskSettings = VoyageAIEmbeddingsTaskSettings.fromMap(taskSettings);
        return new VoyageAIEmbeddingsModel(model, VoyageAIEmbeddingsTaskSettings.of(model.getTaskSettings(), requestTaskSettings));
    }

    public VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        Map<String, Object> serviceSettings,
        Map<String, Object> taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable Map<String, Object> secrets,
        ConfigurationParseContext context
    ) {
        this(
            inferenceId,
            service,
            VoyageAIEmbeddingsServiceSettings.fromMap(serviceSettings, context),
            VoyageAIEmbeddingsTaskSettings.fromMap(taskSettings),
            chunkingSettings,
            DefaultSecretSettings.fromMap(secrets, context)
        );
    }

    /**
     * Builds the request URI for the given model. Contextualized chunk embedding models are served from a dedicated
     * endpoint (see {@link VoyageAIUtils#CONTEXTUALIZED_EMBEDDINGS_PATH}); all other embedding models use the standard
     * embeddings endpoint.
     */
    public static URI buildRequestUri(String modelId) throws URISyntaxException {
        return new URIBuilder().setScheme("https")
            .setHost(HOST)
            .setPathSegments(
                VoyageAIUtils.VERSION_1,
                isContextualizedEmbeddingsModel(modelId) ? VoyageAIUtils.CONTEXTUALIZED_EMBEDDINGS_PATH : VoyageAIUtils.EMBEDDINGS_PATH
            )
            .build();
    }

    /**
     * VoyageAI's contextualized chunk embedding models (the {@code voyage-context-*} family) must be called through the
     * {@code contextualizedembeddings} API rather than the standard {@code embeddings} API.
     */
    public static boolean isContextualizedEmbeddingsModel(String modelId) {
        return modelId != null && modelId.startsWith("voyage-context");
    }

    // should only be used for testing
    VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        String url,
        VoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings
    ) {
        this(inferenceId, service, serviceSettings, taskSettings, chunkingSettings, secretSettings, ServiceUtils.createUri(url));
    }

    private VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        VoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings
    ) {
        this(
            inferenceId,
            service,
            serviceSettings,
            taskSettings,
            chunkingSettings,
            secretSettings,
            buildUri(VoyageAIService.NAME, () -> buildRequestUri(serviceSettings.modelId()))
        );
    }

    private VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        VoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings,
        URI uri
    ) {
        super(
            new ModelConfigurations(inferenceId, TaskType.TEXT_EMBEDDING, service, serviceSettings, taskSettings, chunkingSettings),
            new ModelSecrets(secretSettings),
            uri
        );
    }

    public VoyageAIEmbeddingsModel(ModelConfigurations modelConfigurations, ModelSecrets modelSecrets) {
        super(
            modelConfigurations,
            modelSecrets,
            buildUri(
                VoyageAIService.NAME,
                () -> buildRequestUri(((VoyageAIEmbeddingsServiceSettings) modelConfigurations.getServiceSettings()).modelId())
            )
        );
    }

    private VoyageAIEmbeddingsModel(VoyageAIEmbeddingsModel model, VoyageAIEmbeddingsTaskSettings taskSettings) {
        super(model, taskSettings);
    }

    public VoyageAIEmbeddingsModel(VoyageAIEmbeddingsModel model, VoyageAIEmbeddingsServiceSettings serviceSettings) {
        super(model, serviceSettings);
    }

    @Override
    public VoyageAIEmbeddingsServiceSettings getServiceSettings() {
        return (VoyageAIEmbeddingsServiceSettings) super.getServiceSettings();
    }

    @Override
    public VoyageAIEmbeddingsTaskSettings getTaskSettings() {
        return (VoyageAIEmbeddingsTaskSettings) super.getTaskSettings();
    }

    /**
     * @return {@code true} if this model is a contextualized chunk embedding model and must be called through the
     * contextualized embeddings API.
     */
    public boolean isContextualizedEmbeddingsModel() {
        return isContextualizedEmbeddingsModel(getServiceSettings().modelId());
    }

    @Override
    public ExecutableAction accept(VoyageAIActionVisitor visitor, Map<String, Object> taskSettings) {
        return visitor.create(this, taskSettings);
    }
}
