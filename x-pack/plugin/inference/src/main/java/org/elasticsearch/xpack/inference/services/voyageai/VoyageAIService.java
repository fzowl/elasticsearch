/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai;

import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.util.LazyInitializable;
import org.elasticsearch.core.Strings;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.ChunkInferenceInput;
import org.elasticsearch.inference.ChunkedInference;
import org.elasticsearch.inference.DataFormat;
import org.elasticsearch.inference.DataType;
import org.elasticsearch.inference.EmbeddingInferenceService;
import org.elasticsearch.inference.EmbeddingRequest;
import org.elasticsearch.inference.InferenceServiceConfiguration;
import org.elasticsearch.inference.InferenceServiceExtension;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InferenceString;
import org.elasticsearch.inference.InferenceStringGroup;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.RerankRequest;
import org.elasticsearch.inference.RerankingInferenceService;
import org.elasticsearch.inference.SettingsConfiguration;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.inference.configuration.SettingsConfigurationFieldType;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xpack.core.inference.chunking.EmbeddingRequestChunker;
import org.elasticsearch.xpack.inference.external.action.ExecutableAction;
import org.elasticsearch.xpack.inference.external.http.sender.EmbeddingsInput;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSender;
import org.elasticsearch.xpack.inference.external.http.sender.InferenceInputs;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.services.ModelCreator;
import org.elasticsearch.xpack.inference.services.SenderService;
import org.elasticsearch.xpack.inference.services.ServiceComponents;
import org.elasticsearch.xpack.inference.services.ServiceUtils;
import org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettings;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.voyageai.action.VoyageAIActionCreator;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.BaseVoyageAIEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsModelCreator;
import org.elasticsearch.xpack.inference.services.voyageai.rerank.VoyageAIRerankModel;
import org.elasticsearch.xpack.inference.services.voyageai.rerank.VoyageAIRerankModelCreator;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.external.http.sender.QueryAndDocsInputs.fromRerankRequest;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MODEL_ID;
import static org.elasticsearch.xpack.inference.services.ServiceFields.MULTIMODAL_MODEL;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.createInvalidModelException;
import static org.elasticsearch.xpack.inference.services.ServiceUtils.throwUnsupportedUnifiedCompletionOperation;
import static org.elasticsearch.xpack.inference.services.voyageai.VoyageAIServiceFields.CONTEXTUAL_MODEL;

public class VoyageAIService extends SenderService<VoyageAIModel> implements EmbeddingInferenceService, RerankingInferenceService {
    public static final String NAME = "voyageai";

    private static final String SERVICE_NAME = "Voyage AI by MongoDB";
    private static final EnumSet<TaskType> SUPPORTED_TASK_TYPES = EnumSet.of(TaskType.TEXT_EMBEDDING, TaskType.RERANK, TaskType.EMBEDDING);
    private static final VoyageAIEmbeddingsModelCreator EMBEDDINGS_MODEL_CREATOR = new VoyageAIEmbeddingsModelCreator();
    private static final Map<TaskType, ModelCreator<? extends VoyageAIModel>> MODEL_CREATORS = Map.of(
        TaskType.TEXT_EMBEDDING,
        EMBEDDINGS_MODEL_CREATOR,
        TaskType.EMBEDDING,
        EMBEDDINGS_MODEL_CREATOR,
        TaskType.RERANK,
        new VoyageAIRerankModelCreator()
    );
    private static final Integer DEFAULT_BATCH_SIZE = 7;
    /**
     * The number of inputs per request, chosen so that a request of maximum-length inputs stays within the maximum number of tokens
     * Voyage AI accepts per request (1M for the lite models, 320K for voyage-4 and voyage-3.5, 120K for the others).
     * Voyage AI documents no such limit for voyage-code-4 and the voyage-context models, so they use the default batch size.
     */
    private static final Map<String, Integer> MODEL_BATCH_SIZES = Map.ofEntries(
        Map.entry("voyage-4-large", 7),
        Map.entry("voyage-4", 10),
        Map.entry("voyage-4-lite", 30),
        Map.entry("voyage-code-4", 7),
        Map.entry("voyage-context-4", 7),
        Map.entry("voyage-context-3", 7),
        Map.entry("voyage-multimodal-3", 7),
        Map.entry("voyage-multimodal-3.5", 7),
        Map.entry("voyage-3-large", 7),
        Map.entry("voyage-3.5", 10),
        Map.entry("voyage-3.5-lite", 30),
        Map.entry("voyage-code-3", 7),
        Map.entry("voyage-3", 10),
        Map.entry("voyage-3-lite", 30),
        Map.entry("voyage-finance-2", 7),
        Map.entry("voyage-law-2", 7),
        Map.entry("voyage-code-2", 7),
        Map.entry("voyage-2", 72),
        Map.entry("voyage-02", 72)
    );

    private static final Map<String, Integer> RERANKERS_INPUT_SIZE = Map.of(
        "rerank-lite-1",
        2800 // The smallest model has a 4K context length https://docs.voyageai.com/docs/reranker
    );

    /**
     * Apart from rerank-lite-1 all other models have a context length of at least 8k.
     * This value is based on 1 token == 0.75 words and allowing for some overhead
     */
    private static final int DEFAULT_RERANKER_INPUT_SIZE_WORDS = 5500;

    public static final EnumSet<InputType> VALID_INPUT_TYPE_VALUES = EnumSet.of(
        InputType.INGEST,
        InputType.SEARCH,
        InputType.INTERNAL_INGEST,
        InputType.INTERNAL_SEARCH
    );

    public static final TransportVersion VOYAGE_AI_INTEGRATION_ADDED = TransportVersion.fromName("voyage_ai_integration_added");
    public static final TransportVersion VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED = TransportVersion.fromName(
        "voyage_ai_multimodal_embeddings_added"
    );
    /**
     * The transport version that added the {@code contextual_model} flag to the {@code text_embedding} service settings. The transport
     * version tooling only allows one new transport version for each change, so for now this is the multimodal one. When the contextual
     * embeddings are split out into their own change, replace this with
     * {@code TransportVersion.fromName("voyage_ai_contextual_embeddings_added")} and run {@code ./gradlew generateTransportVersion}.
     */
    public static final TransportVersion VOYAGE_AI_CONTEXTUAL_EMBEDDINGS_ADDED = VOYAGE_AI_MULTIMODAL_EMBEDDINGS_ADDED;

    public VoyageAIService(
        HttpRequestSender.Factory factory,
        ServiceComponents serviceComponents,
        InferenceServiceExtension.InferenceServiceFactoryContext context
    ) {
        this(factory, serviceComponents, context.clusterService());
    }

    public VoyageAIService(HttpRequestSender.Factory factory, ServiceComponents serviceComponents, ClusterService clusterService) {
        super(factory, serviceComponents, clusterService, MODEL_CREATORS);
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public InferenceServiceConfiguration getConfiguration() {
        return Configuration.get();
    }

    @Override
    public EnumSet<TaskType> supportedTaskTypes() {
        return SUPPORTED_TASK_TYPES;
    }

    @Override
    protected void doUnifiedCompletionInfer(
        Model model,
        UnifiedChatInput inputs,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    ) {
        throwUnsupportedUnifiedCompletionOperation(NAME);
    }

    @Override
    public void doInfer(
        Model model,
        InferenceInputs inputs,
        Map<String, Object> taskSettings,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    ) {
        if (model instanceof VoyageAIModel == false) {
            listener.onFailure(createInvalidModelException(model));
            return;
        }

        VoyageAIModel voyageaiModel = (VoyageAIModel) model;
        var actionCreator = new VoyageAIActionCreator(getSender(), getServiceComponents());

        var action = voyageaiModel.accept(actionCreator, taskSettings);
        action.execute(inputs, timeout, listener);
    }

    @Override
    protected void validateInputType(InputType inputType, Model model, ValidationException validationException) {
        ServiceUtils.validateInputTypeAgainstAllowlist(inputType, VALID_INPUT_TYPE_VALUES, SERVICE_NAME, validationException);
    }

    @Override
    protected void doRerankInfer(Model model, RerankRequest request, TimeValue timeout, ActionListener<InferenceServiceResults> listener) {
        if (!(model instanceof VoyageAIRerankModel voyageAIRerankModel)) {
            listener.onFailure(createInvalidModelException(model));
            return;
        }
        var actionCreator = new VoyageAIActionCreator(getSender(), getServiceComponents());

        var action = voyageAIRerankModel.accept(actionCreator, request.taskSettings());
        action.execute(fromRerankRequest(request), timeout, listener);
    }

    @Override
    protected void doChunkedInfer(
        Model model,
        List<ChunkInferenceInput> inputs,
        Map<String, Object> taskSettings,
        InputType inputType,
        TimeValue timeout,
        ActionListener<List<ChunkedInference>> listener
    ) {
        if (model instanceof VoyageAIModel == false) {
            listener.onFailure(createInvalidModelException(model));
            return;
        }

        VoyageAIModel voyageaiModel = (VoyageAIModel) model;
        var actionCreator = new VoyageAIActionCreator(getSender(), getServiceComponents());

        List<EmbeddingRequestChunker.BatchRequestAndListener> batchedRequests = new EmbeddingRequestChunker<>(
            inputs,
            getBatchSize(voyageaiModel),
            getRegexReadLimitFactor(),
            voyageaiModel.getConfigurations().getChunkingSettings()
        ).batchRequestsWithListeners(listener);

        for (var request : batchedRequests) {
            var action = voyageaiModel.accept(actionCreator, taskSettings);
            action.execute(
                new EmbeddingsInput(request.batch().inputs(), request.batch().ramBytesUsed(), inputType),
                timeout,
                request.listener()
            );
        }
    }

    @Override
    protected void doEmbeddingInfer(
        Model model,
        EmbeddingRequest request,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    ) {
        if (model instanceof VoyageAIEmbeddingsModel embeddingsModel) {
            var validationError = validateEmbeddingInputs(embeddingsModel.getServiceSettings().isMultimodal(), request.inputs());
            if (validationError != null) {
                listener.onFailure(new ElasticsearchStatusException(validationError, RestStatus.BAD_REQUEST));
                return;
            }
            var actionCreator = new VoyageAIActionCreator(getSender(), getServiceComponents());

            ExecutableAction action = embeddingsModel.accept(actionCreator, request.taskSettings());
            action.execute(new EmbeddingsInput(request.inputs(), request.inputType()), timeout, listener);
        } else {
            listener.onFailure(createInvalidModelException(model));
        }
    }

    /**
     * Voyage AI only supports text and base64 image inputs, and only multimodal models accept anything other than a single text item
     * for each input.
     *
     * @return a description of what is not supported by the model, or {@code null} if the inputs can be sent to it
     */
    private static String validateEmbeddingInputs(boolean multimodalModel, List<InferenceStringGroup> inputs) {
        if (multimodalModel == false) {
            if (InferenceStringGroup.containsNonTextEntry(inputs)) {
                return "Non-text input provided for text-only model";
            }
            var indexOfMultipleItems = InferenceStringGroup.indexContainingMultipleInferenceStrings(inputs);
            if (indexOfMultipleItems != null) {
                return Strings.format(
                    "Field [%1$s] must contain a single item for a text-only model. "
                        + "[%1$s] object with multiple items found at $.%2$s.%1$s[%3$d]",
                    InferenceStringGroup.CONTENT_FIELD,
                    EmbeddingRequest.INPUT_FIELD,
                    indexOfMultipleItems
                );
            }
            return null;
        }
        for (var inferenceStringGroup : inputs) {
            for (InferenceString inferenceString : inferenceStringGroup.inferenceStrings()) {
                var supported = inferenceString.dataType() == DataType.TEXT
                    || (inferenceString.dataType() == DataType.IMAGE && inferenceString.dataFormat() == DataFormat.BASE64);
                if (supported == false) {
                    return Strings.format(
                        "The [%s] service only supports text and base64 image inputs for embeddings, but received an input of type [%s] "
                            + "and format [%s]",
                        NAME,
                        inferenceString.dataType(),
                        inferenceString.dataFormat()
                    );
                }
            }
        }
        return null;
    }

    @Override
    public boolean supportsNonTextEmbeddingContent() {
        return true;
    }

    @Override
    protected boolean supportsMultipleItemsPerContent() {
        return true;
    }

    @Override
    public boolean requiresSingleInputEmbeddingRequest(Model model, InferenceString input) {
        return false;
    }

    private static int getBatchSize(VoyageAIModel model) {
        return MODEL_BATCH_SIZES.getOrDefault(model.getServiceSettings().modelId(), DEFAULT_BATCH_SIZE);
    }

    @Override
    public Model updateModelWithEmbeddingDetails(Model model, int embeddingSize) {
        if (model instanceof VoyageAIEmbeddingsModel embeddingsModel) {
            var serviceSettings = embeddingsModel.getServiceSettings();
            var similarityToUse = Objects.requireNonNullElse(
                serviceSettings.similarity(),
                defaultSimilarity(serviceSettings.embeddingType())
            );

            var updatedServiceSettings = BaseVoyageAIEmbeddingsServiceSettings.updateEmbeddingDetails(
                serviceSettings,
                embeddingSize,
                similarityToUse
            );

            if (updatedServiceSettings.equals(serviceSettings)) {
                return model;
            }

            return new VoyageAIEmbeddingsModel(embeddingsModel, updatedServiceSettings);
        } else {
            throw ServiceUtils.invalidModelTypeForUpdateModelWithEmbeddingDetails(model.getClass());
        }
    }

    /**
     * Return the default similarity measure for the embedding type.
     * Float embeddings default to cosine similarity since we cannot guarantee third-party providers
     * return unit-length vectors. Non-float types (INT8, BYTE, BIT, BINARY) keep their existing
     * dot_product default.
     *
     * @param embeddingType the VoyageAI embedding type
     * @return The default similarity.
     */
    static SimilarityMeasure defaultSimilarity(VoyageAIEmbeddingType embeddingType) {
        return switch (embeddingType) {
            case FLOAT -> SimilarityMeasure.COSINE;
            case INT8, BYTE, BIT, BINARY -> SimilarityMeasure.DOT_PRODUCT;
        };
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return VOYAGE_AI_INTEGRATION_ADDED;
    }

    @Override
    public int rerankerWindowSize(String modelId) {
        Integer inputSize = RERANKERS_INPUT_SIZE.get(modelId);
        return inputSize != null ? inputSize : DEFAULT_RERANKER_INPUT_SIZE_WORDS;
    }

    public static class Configuration {
        public static InferenceServiceConfiguration get() {
            return CONFIGURATION.getOrCompute();
        }

        private static final LazyInitializable<InferenceServiceConfiguration, RuntimeException> CONFIGURATION = new LazyInitializable<>(
            () -> {
                var configurationMap = new HashMap<String, SettingsConfiguration>();

                configurationMap.put(
                    MODEL_ID,
                    new SettingsConfiguration.Builder(SUPPORTED_TASK_TYPES).setDescription(
                        "The name of the model to use for the inference task."
                    )
                        .setLabel("Model ID")
                        .setRequired(true)
                        .setSensitive(false)
                        .setUpdatable(false)
                        .setType(SettingsConfigurationFieldType.STRING)
                        .build()
                );

                configurationMap.put(
                    MULTIMODAL_MODEL,
                    new SettingsConfiguration.Builder(EnumSet.of(TaskType.EMBEDDING)).setDescription(
                        "Whether the model accepts multimodal inputs, which are sent to the Voyage AI multimodal embeddings API."
                    )
                        .setLabel("Multimodal model")
                        .setDefaultValue(true)
                        .setRequired(false)
                        .setSensitive(false)
                        .setUpdatable(false)
                        .setType(SettingsConfigurationFieldType.BOOLEAN)
                        .build()
                );

                configurationMap.put(
                    CONTEXTUAL_MODEL,
                    new SettingsConfiguration.Builder(EnumSet.of(TaskType.TEXT_EMBEDDING)).setDescription(
                        "Whether the model is a contextualized embeddings model, such as voyage-context-3, which is served by the "
                            + "Voyage AI contextualized embeddings API."
                    )
                        .setLabel("Contextual model")
                        .setDefaultValue(false)
                        .setRequired(false)
                        .setSensitive(false)
                        .setUpdatable(false)
                        .setType(SettingsConfigurationFieldType.BOOLEAN)
                        .build()
                );

                configurationMap.putAll(DefaultSecretSettings.toSettingsConfiguration(SUPPORTED_TASK_TYPES));
                configurationMap.putAll(RateLimitSettings.toSettingsConfiguration(SUPPORTED_TASK_TYPES));

                return new InferenceServiceConfiguration.Builder().setService(NAME)
                    .setName(SERVICE_NAME)
                    .setTaskTypes(SUPPORTED_TASK_TYPES)
                    .setConfigurations(configurationMap)
                    .build();
            }
        );
    }
}
