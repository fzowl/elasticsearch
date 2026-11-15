/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.is;

public class VoyageAIUtilsTests extends ESTestCase {

    public void testCreateRequestSourceHeader() {
        var requestSourceHeader = VoyageAIUtils.createRequestSourceHeader();

        assertThat(requestSourceHeader.getName(), is("Request-Source"));
        assertThat(requestSourceHeader.getValue(), is("unspecified:elasticsearch"));
    }

    public void testResolveHost_UsesVoyageHost_ForVoyageApiKey() {
        assertThat(VoyageAIUtils.resolveHost(new SecureString("pa-key".toCharArray())), is("api.voyageai.com"));
    }

    public void testResolveHost_UsesMongoDbHost_ForMongoDbApiKey() {
        assertThat(VoyageAIUtils.resolveHost(new SecureString("al-key".toCharArray())), is("ai.mongodb.com"));
    }

    public void testResolveHost_UsesVoyageHost_WhenPrefixIsNotAtTheStartOrKeyIsTooShort() {
        assertThat(VoyageAIUtils.resolveHost(new SecureString("xal-key".toCharArray())), is("api.voyageai.com"));
        assertThat(VoyageAIUtils.resolveHost(new SecureString("a".toCharArray())), is("api.voyageai.com"));
        assertThat(VoyageAIUtils.resolveHost(new SecureString("".toCharArray())), is("api.voyageai.com"));
        assertThat(VoyageAIUtils.resolveHost(null), is("api.voyageai.com"));
    }
}
