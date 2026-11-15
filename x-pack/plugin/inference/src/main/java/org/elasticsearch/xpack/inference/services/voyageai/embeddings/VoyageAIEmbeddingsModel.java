/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.embeddings;

import org.apache.hc.core5.net.URIBuilder;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.ServiceSettings;
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
        ConfigurationParseContext context,
        TaskType taskType
    ) {
        this(
            inferenceId,
            service,
            createServiceSettings(serviceSettings, taskType, context),
            VoyageAIEmbeddingsTaskSettings.fromMap(taskSettings),
            chunkingSettings,
            DefaultSecretSettings.fromMap(secrets, context),
            taskType
        );
    }

    private VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings,
        TaskType taskType
    ) {
        this(
            inferenceId,
            service,
            serviceSettings,
            taskSettings,
            chunkingSettings,
            secretSettings,
            buildUriFromSettings(serviceSettings, secretSettings),
            taskType
        );
    }

    private static URI buildRequestUri(@Nullable DefaultSecretSettings secretSettings, String path) throws URISyntaxException {
        return new URIBuilder().setScheme("https")
            .setHost(VoyageAIUtils.resolveHost(secretSettings == null ? null : secretSettings.apiKey()))
            .setPathSegments(VoyageAIUtils.VERSION_1, path)
            .build();
    }

    /**
     * Builds the URI of the endpoint that serves the model: the multimodal embeddings, contextualized embeddings or embeddings endpoint,
     * depending on the service settings, on the host that serves the API key. The endpoint is never inferred from the model id.
     */
    static URI buildUriFromSettings(ServiceSettings serviceSettings, @Nullable DefaultSecretSettings secretSettings) {
        final String path;
        if (serviceSettings.isMultimodal()) {
            path = VoyageAIUtils.MULTIMODAL_EMBEDDINGS_PATH;
        } else if (serviceSettings instanceof VoyageAITextEmbeddingServiceSettings textSettings && textSettings.isContextual()) {
            path = VoyageAIUtils.CONTEXTUALIZED_EMBEDDINGS_PATH;
        } else {
            path = VoyageAIUtils.EMBEDDINGS_PATH;
        }
        return buildUri(VoyageAIService.NAME, () -> buildRequestUri(secretSettings, path));
    }

    // should only be used for testing
    VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        String url,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings,
        TaskType taskType
    ) {
        this(inferenceId, service, serviceSettings, taskSettings, chunkingSettings, secretSettings, ServiceUtils.createUri(url), taskType);
    }

    private VoyageAIEmbeddingsModel(
        String inferenceId,
        String service,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        ChunkingSettings chunkingSettings,
        @Nullable DefaultSecretSettings secretSettings,
        URI uri,
        TaskType taskType
    ) {
        super(
            new ModelConfigurations(inferenceId, taskType, service, serviceSettings, taskSettings, chunkingSettings),
            new ModelSecrets(secretSettings),
            uri
        );
    }

    public VoyageAIEmbeddingsModel(ModelConfigurations modelConfigurations, ModelSecrets modelSecrets) {
        super(
            modelConfigurations,
            modelSecrets,
            buildUriFromSettings(modelConfigurations.getServiceSettings(), secretSettingsOf(modelSecrets))
        );
    }

    private VoyageAIEmbeddingsModel(VoyageAIEmbeddingsModel model, VoyageAIEmbeddingsTaskSettings taskSettings) {
        super(model, taskSettings);
    }

    public VoyageAIEmbeddingsModel(VoyageAIEmbeddingsModel model, BaseVoyageAIEmbeddingsServiceSettings serviceSettings) {
        super(model, serviceSettings);
    }

    @Override
    public BaseVoyageAIEmbeddingsServiceSettings getServiceSettings() {
        return (BaseVoyageAIEmbeddingsServiceSettings) super.getServiceSettings();
    }

    @Override
    public VoyageAIEmbeddingsTaskSettings getTaskSettings() {
        return (VoyageAIEmbeddingsTaskSettings) super.getTaskSettings();
    }

    @Override
    public ExecutableAction accept(VoyageAIActionVisitor visitor, Map<String, Object> taskSettings) {
        return visitor.create(this, taskSettings);
    }

    private static BaseVoyageAIEmbeddingsServiceSettings createServiceSettings(
        Map<String, Object> serviceSettings,
        TaskType taskType,
        ConfigurationParseContext context
    ) {
        return switch (taskType) {
            case TEXT_EMBEDDING -> VoyageAITextEmbeddingServiceSettings.fromMap(serviceSettings, context);
            case EMBEDDING -> VoyageAIEmbeddingServiceSettings.fromMap(serviceSettings, context);
            default -> throw new IllegalArgumentException("Unsupported task type for Voyage AI embeddings: " + taskType);
        };
    }
}
