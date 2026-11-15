/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.apache.hc.core5.http.Header;
import org.apache.hc.core5.http.message.BasicHeader;
import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.core.Nullable;

public class VoyageAIUtils {
    public static final String HOST = "api.voyageai.com";
    /**
     * Voyage AI is now part of MongoDB. API keys issued by the MongoDB Atlas platform start with {@link #MONGODB_API_KEY_PREFIX}
     * and are served from {@link #MONGODB_HOST}, as in the official Voyage AI clients.
     */
    public static final String MONGODB_HOST = "ai.mongodb.com";
    public static final String MONGODB_API_KEY_PREFIX = "al-";
    public static final String VERSION_1 = "v1";
    public static final String EMBEDDINGS_PATH = "embeddings";
    public static final String MULTIMODAL_EMBEDDINGS_PATH = "multimodalembeddings";
    public static final String CONTEXTUALIZED_EMBEDDINGS_PATH = "contextualizedembeddings";
    public static final String RERANK_PATH = "rerank";
    /** The chunk size, in tokens, requested from Voyage AI when it chunks documents for the contextualized embeddings API. */
    public static final int CONTEXTUALIZED_CHUNK_SIZE = 32_000;
    public static final String REQUEST_SOURCE_HEADER = "Request-Source";
    public static final String ELASTIC_REQUEST_SOURCE = "unspecified:elasticsearch";

    public static Header createRequestSourceHeader() {
        return new BasicHeader(REQUEST_SOURCE_HEADER, ELASTIC_REQUEST_SOURCE);
    }

    /**
     * @param apiKey the API key of the inference endpoint, which may be null when the secrets are not available
     * @return the host to send requests to for the given API key
     */
    public static String resolveHost(@Nullable SecureString apiKey) {
        if (apiKey == null || apiKey.length() < MONGODB_API_KEY_PREFIX.length()) {
            return HOST;
        }
        for (int i = 0; i < MONGODB_API_KEY_PREFIX.length(); i++) {
            if (apiKey.charAt(i) != MONGODB_API_KEY_PREFIX.charAt(i)) {
                return HOST;
            }
        }
        return MONGODB_HOST;
    }

    private VoyageAIUtils() {}
}
