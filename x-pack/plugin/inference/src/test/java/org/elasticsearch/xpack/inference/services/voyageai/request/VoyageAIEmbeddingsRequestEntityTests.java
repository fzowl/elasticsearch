/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.voyageai.request;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.inference.DataFormat;
import org.elasticsearch.inference.DataType;
import org.elasticsearch.inference.InferenceString;
import org.elasticsearch.inference.InferenceStringGroup;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.BaseVoyageAIEmbeddingsServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingServiceSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingType;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAIEmbeddingsTaskSettings;
import org.elasticsearch.xpack.inference.services.voyageai.embeddings.VoyageAITextEmbeddingServiceSettings;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

public class VoyageAIEmbeddingsRequestEntityTests extends ESTestCase {

    /**
     * Voyage AI requires images to be base64 data URIs with a valid image MIME type
     */
    public static final String TEST_IMAGE_DATA_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD";
    public static final String TEST_URL_IMAGE = "https://example.com/image.jpg";

    private static final String TEST_INPUT = "some input";
    private static final String TEST_MODEL_ID = "some-model-id";
    private static final int TEST_RATE_LIMIT = 10;
    private static final int TEST_DIMENSIONS = 2048;
    private static final int TEST_MAX_INPUT_TOKENS = 1024;

    public void testXContent_WritesAllFields_Float() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        assertToXContent_WritesAllFields(
            InputType.INTERNAL_SEARCH,
            new VoyageAITextEmbeddingServiceSettings(
                TEST_MODEL_ID,
                new RateLimitSettings(TEST_RATE_LIMIT),
                embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                false
            ),
            new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null),
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "input_type": "query",
                    "output_dimension": %d,
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString())
        );
    }

    public void testXContent_WritesAllFields_Int8() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.INT8;
        assertToXContent_WritesAllFields(
            InputType.INGEST,
            new VoyageAITextEmbeddingServiceSettings(
                TEST_MODEL_ID,
                new RateLimitSettings(TEST_RATE_LIMIT),
                embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                false
            ),
            new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null),
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "input_type": "document",
                    "output_dimension": %d,
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString())
        );
    }

    public void testXContent_WritesAllFields_Binary() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.BINARY;
        assertToXContent_WritesAllFields(
            InputType.INTERNAL_SEARCH,
            new VoyageAITextEmbeddingServiceSettings(
                TEST_MODEL_ID,
                new RateLimitSettings(TEST_RATE_LIMIT),
                embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                false
            ),
            new VoyageAIEmbeddingsTaskSettings(null, null),
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "input_type": "query",
                    "output_dimension": %d,
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString())
        );
    }

    public void testXContent_FallsBackToTaskSettingsInputType_WhenRootInputTypeIsNull() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        assertToXContent_WritesAllFields(
            null,
            new VoyageAITextEmbeddingServiceSettings(
                TEST_MODEL_ID,
                new RateLimitSettings(TEST_RATE_LIMIT),
                embeddingType,
                SimilarityMeasure.DOT_PRODUCT,
                TEST_DIMENSIONS,
                TEST_MAX_INPUT_TOKENS,
                false
            ),
            new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null),
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "input_type": "document",
                    "output_dimension": %d,
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType.toString())
        );
    }

    public void testXContent_WritesNoOptionalFields_WhenTheyAreNotDefined() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.FLOAT;
        assertToXContent_WritesAllFields(
            null,
            new VoyageAITextEmbeddingServiceSettings(TEST_MODEL_ID, null, embeddingType, null, null, null, false),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, embeddingType.toString())
        );
    }

    public void testXContent_WritesAllFields_NotContextualOrMultimodalWhenModelIdLooksLikeOne() throws IOException {
        // The format of the request is chosen by the settings and never by the model id
        assertToXContent_WritesAllFields(
            InputType.INGEST,
            new VoyageAITextEmbeddingServiceSettings("voyage-context-4", null, VoyageAIEmbeddingType.FLOAT, null, null, null, false),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "voyage-context-4",
                    "input_type": "document",
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
        assertToXContent_WritesAllFields(
            InputType.INGEST,
            new VoyageAITextEmbeddingServiceSettings("voyage-multimodal-3.5", null, VoyageAIEmbeddingType.FLOAT, null, null, null, false),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "voyage-multimodal-3.5",
                    "input_type": "document",
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
    }

    public void testXContent_Contextualized_WritesFlatInputsAndAutoChunking_ForDocuments() throws IOException {
        assertToXContent_WritesAllFields(
            InputType.INGEST,
            contextualSettings("voyage-context-4", null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": ["%s"],
                    "enable_auto_chunking": true,
                    "chunk_size": 32000,
                    "model": "voyage-context-4",
                    "input_type": "document",
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
    }

    public void testXContent_Contextualized_WritesAutoChunking_WhenDocumentInputTypeIsInTheTaskSettings() throws IOException {
        assertToXContent_WritesAllFields(
            null,
            contextualSettings("voyage-context-3", null),
            new VoyageAIEmbeddingsTaskSettings(InputType.INGEST, null),
            Strings.format("""
                {
                    "inputs": ["%s"],
                    "enable_auto_chunking": true,
                    "chunk_size": 32000,
                    "model": "voyage-context-3",
                    "input_type": "document",
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
    }

    public void testXContent_Contextualized_DoesNotEnableAutoChunking_ForQueries() throws IOException {
        assertToXContent_WritesAllFields(
            InputType.SEARCH,
            contextualSettings("voyage-context-4", 256),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": ["%s"],
                    "model": "voyage-context-4",
                    "input_type": "query",
                    "output_dimension": 256,
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
    }

    public void testXContent_Contextualized_DoesNotEnableAutoChunking_WithoutInputType() throws IOException {
        assertToXContent_WritesAllFields(
            null,
            contextualSettings("voyage-context-4", null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": ["%s"],
                    "model": "voyage-context-4",
                    "output_dtype": "float"
                }
                """, TEST_INPUT)
        );
    }

    public void testXContent_Contextualized_WritesOneFlatInputForEachInputGroup() throws IOException {
        assertToXContent(
            List.of(new InferenceStringGroup("first"), new InferenceStringGroup("second")),
            InputType.SEARCH,
            contextualSettings("voyage-context-4", null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            """
                {
                    "inputs": ["first", "second"],
                    "model": "voyage-context-4",
                    "input_type": "query",
                    "output_dtype": "float"
                }
                """
        );
    }

    public void testXContent_Multimodal_WritesAllFields_ForTextAndImageInGroup() throws IOException {
        var embeddingType = VoyageAIEmbeddingType.INT8;
        assertToXContent(
            List.of(
                new InferenceStringGroup(
                    List.of(InferenceString.ofText(TEST_INPUT), new InferenceString(DataType.IMAGE, DataFormat.BASE64, TEST_IMAGE_DATA_URI))
                )
            ),
            InputType.INTERNAL_INGEST,
            multimodalSettings(embeddingType, TEST_DIMENSIONS),
            new VoyageAIEmbeddingsTaskSettings(InputType.SEARCH, true),
            Strings.format("""
                {
                    "inputs": [
                        {
                            "content": [
                                {"type": "text", "text": "%s"},
                                {"type": "image_base64", "image_base64": "%s"}
                            ]
                        }
                    ],
                    "model": "%s",
                    "input_type": "document",
                    "truncation": true,
                    "output_dimension": %d,
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_IMAGE_DATA_URI, TEST_MODEL_ID, TEST_DIMENSIONS, embeddingType)
        );
    }

    public void testXContent_Multimodal_WritesAnInputObjectForEachGroup() throws IOException {
        assertToXContent(
            List.of(
                new InferenceStringGroup("first text"),
                new InferenceStringGroup(new InferenceString(DataType.IMAGE, DataFormat.BASE64, TEST_IMAGE_DATA_URI)),
                new InferenceStringGroup(
                    List.of(
                        new InferenceString(DataType.IMAGE, DataFormat.BASE64, TEST_IMAGE_DATA_URI),
                        InferenceString.ofText("second text"),
                        new InferenceString(DataType.IMAGE, DataFormat.BASE64, TEST_IMAGE_DATA_URI)
                    )
                )
            ),
            InputType.SEARCH,
            multimodalSettings(VoyageAIEmbeddingType.FLOAT, null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": [
                        {"content": [{"type": "text", "text": "first text"}]},
                        {"content": [{"type": "image_base64", "image_base64": "%1$s"}]},
                        {
                            "content": [
                                {"type": "image_base64", "image_base64": "%1$s"},
                                {"type": "text", "text": "second text"},
                                {"type": "image_base64", "image_base64": "%1$s"}
                            ]
                        }
                    ],
                    "model": "%2$s",
                    "input_type": "query",
                    "output_dtype": "float"
                }
                """, TEST_IMAGE_DATA_URI, TEST_MODEL_ID)
        );
    }

    public void testXContent_Multimodal_OnlyWritesTheRequiredFieldsAndOutputDtype_WhenOptionalFieldsAreNotDefined() throws IOException {
        assertToXContent(
            List.of(new InferenceStringGroup(TEST_INPUT)),
            null,
            multimodalSettings(VoyageAIEmbeddingType.FLOAT, null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": [{"content": [{"type": "text", "text": "%s"}]}],
                    "model": "%s",
                    "output_dtype": "float"
                }
                """, TEST_INPUT, TEST_MODEL_ID)
        );
    }

    public void testXContent_Multimodal_WritesOutputDtype_WhenItIsNotTheDefault() throws IOException {
        var embeddingType = randomValueOtherThan(VoyageAIEmbeddingType.FLOAT, () -> randomFrom(VoyageAIEmbeddingType.values()));
        assertToXContent(
            List.of(new InferenceStringGroup(TEST_INPUT)),
            null,
            multimodalSettings(embeddingType, null),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "inputs": [{"content": [{"type": "text", "text": "%s"}]}],
                    "model": "%s",
                    "output_dtype": "%s"
                }
                """, TEST_INPUT, TEST_MODEL_ID, embeddingType.toRequestString())
        );
    }

    public void testXContent_EmbeddingTaskSettings_WithMultimodalModelFalse_WritesTextInputs() throws IOException {
        assertToXContent(
            List.of(new InferenceStringGroup(TEST_INPUT)),
            null,
            new VoyageAIEmbeddingServiceSettings(
                TEST_MODEL_ID,
                null,
                VoyageAIEmbeddingType.FLOAT,
                SimilarityMeasure.DOT_PRODUCT,
                null,
                null,
                false,
                false
            ),
            VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS,
            Strings.format("""
                {
                    "input": ["%s"],
                    "model": "%s",
                    "output_dtype": "float"
                }
                """, TEST_INPUT, TEST_MODEL_ID)
        );
    }

    public void testXContent_Multimodal_ThrowsForUrlImage() {
        var exception = expectThrows(
            IllegalArgumentException.class,
            () -> toXContent(
                List.of(new InferenceStringGroup(new InferenceString(DataType.IMAGE, DataFormat.URL, TEST_URL_IMAGE))),
                null,
                multimodalSettings(VoyageAIEmbeddingType.FLOAT, null),
                VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS
            )
        );

        assertThat(exception.getMessage(), containsString("only text and base64 image inputs are supported"));
    }

    public void testXContent_Multimodal_ThrowsForInputTypesOtherThanTextAndImage() {
        var dataType = randomFrom(DataType.AUDIO, DataType.VIDEO, DataType.PDF);
        var exception = expectThrows(
            IllegalArgumentException.class,
            () -> toXContent(
                List.of(new InferenceStringGroup(new InferenceString(dataType, DataFormat.BASE64, TEST_IMAGE_DATA_URI))),
                null,
                multimodalSettings(VoyageAIEmbeddingType.FLOAT, null),
                VoyageAIEmbeddingsTaskSettings.EMPTY_SETTINGS
            )
        );

        assertThat(exception.getMessage(), containsString("only text and base64 image inputs are supported"));
    }

    private static VoyageAITextEmbeddingServiceSettings contextualSettings(String modelId, Integer dimensions) {
        return new VoyageAITextEmbeddingServiceSettings(
            modelId,
            null,
            VoyageAIEmbeddingType.FLOAT,
            null,
            dimensions,
            null,
            dimensions != null,
            true
        );
    }

    private static VoyageAIEmbeddingServiceSettings multimodalSettings(VoyageAIEmbeddingType embeddingType, Integer dimensions) {
        return new VoyageAIEmbeddingServiceSettings(
            TEST_MODEL_ID,
            new RateLimitSettings(TEST_RATE_LIMIT),
            embeddingType,
            SimilarityMeasure.DOT_PRODUCT,
            dimensions,
            TEST_MAX_INPUT_TOKENS,
            dimensions != null,
            true
        );
    }

    private static void assertToXContent_WritesAllFields(
        InputType inputType,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        String expectedResult
    ) throws IOException {
        assertToXContent(List.of(new InferenceStringGroup(TEST_INPUT)), inputType, serviceSettings, taskSettings, expectedResult);
    }

    private static void assertToXContent(
        List<InferenceStringGroup> input,
        InputType inputType,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings,
        String expectedResult
    ) throws IOException {
        assertThat(toXContent(input, inputType, serviceSettings, taskSettings), is(XContentHelper.stripWhitespace(expectedResult)));
    }

    private static String toXContent(
        List<InferenceStringGroup> input,
        InputType inputType,
        BaseVoyageAIEmbeddingsServiceSettings serviceSettings,
        VoyageAIEmbeddingsTaskSettings taskSettings
    ) throws IOException {
        var entity = new VoyageAIEmbeddingsRequestEntity(input, inputType, serviceSettings, taskSettings);

        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        entity.toXContent(builder, null);
        return Strings.toString(builder);
    }
}
