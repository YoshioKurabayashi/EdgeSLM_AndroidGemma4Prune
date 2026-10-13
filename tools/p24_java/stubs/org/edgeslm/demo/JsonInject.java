package org.edgeslm.demo;

/** コンパイルの確かめ用（アプリには入れない）。実物は Kotlin の `object JsonInject` */
public final class JsonInject {
    public static final JsonInject INSTANCE = new JsonInject();
    private JsonInject() {}
    public final String addRuntimePaths(String jobJson, String baseDir, String nativeLibDir, String overridesJson) { return jobJson; }
}
