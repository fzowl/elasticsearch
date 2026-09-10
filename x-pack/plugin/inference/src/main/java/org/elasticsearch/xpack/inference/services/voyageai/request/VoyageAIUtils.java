/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.apache.http.Header;
import org.apache.http.message.BasicHeader;
import org.elasticsearch.core.Nullable;

public class VoyageAIUtils {
    public static final String HOST = "api.voyageai.com";
    public static final String MONGODB_HOST = "ai.mongodb.com";
    /**
     * API keys issued through MongoDB (VoyageAI by MongoDB) are prefixed with {@code al-} and must be routed to the
     * MongoDB-hosted endpoint rather than the default VoyageAI host.
     */
    public static final String MONGODB_API_KEY_PREFIX = "al-";
    public static final String VERSION_1 = "v1";
    public static final String EMBEDDINGS_PATH = "embeddings";
    public static final String CONTEXTUALIZED_EMBEDDINGS_PATH = "contextualizedembeddings";
    public static final String RERANK_PATH = "rerank";
    public static final String REQUEST_SOURCE_HEADER = "Request-Source";
    public static final String ELASTIC_REQUEST_SOURCE = "unspecified:elasticsearch";

    public static Header createRequestSourceHeader() {
        return new BasicHeader(REQUEST_SOURCE_HEADER, ELASTIC_REQUEST_SOURCE);
    }

    /**
     * Resolves the API host for the given API key, mirroring the official {@code voyageai-python} client's
     * {@code get_default_base_url()} (see
     * <a href="https://github.com/voyage-ai/voyageai-python/blob/9aca465efd0011c478031f6d584b70a3a4393a7c/voyageai/util.py#L99">
     * util.py</a>): keys prefixed with {@link #MONGODB_API_KEY_PREFIX} are served from {@link #MONGODB_HOST}
     * ({@code https://ai.mongodb.com/v1}); all other keys use the default {@link #HOST}
     * ({@code https://api.voyageai.com/v1}).
     */
    public static String resolveHost(@Nullable CharSequence apiKey) {
        return isMongoDBApiKey(apiKey) ? MONGODB_HOST : HOST;
    }

    /**
     * @return {@code true} if the API key is a MongoDB-issued key (prefixed with {@link #MONGODB_API_KEY_PREFIX}). The
     * check reads only the prefix characters so a {@code SecureString} is not fully materialized.
     */
    public static boolean isMongoDBApiKey(@Nullable CharSequence apiKey) {
        if (apiKey == null || apiKey.length() < MONGODB_API_KEY_PREFIX.length()) {
            return false;
        }
        for (int i = 0; i < MONGODB_API_KEY_PREFIX.length(); i++) {
            if (apiKey.charAt(i) != MONGODB_API_KEY_PREFIX.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private VoyageAIUtils() {}
}
