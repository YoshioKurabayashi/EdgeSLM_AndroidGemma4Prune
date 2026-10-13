package org.edgeslm.demo;

/** コンパイルの確かめ用（アプリには入れない）。実物は Kotlin の `object NativeBridge` */
public final class NativeBridge {
    public static final NativeBridge INSTANCE = new NativeBridge();
    private NativeBridge() {}
    public final native long nativeCreate(String configJson);
    public final native String nativeDescribe(long handle);
    public final native String nativeRunJob(long handle, String jobJson);
    public final native String nativeProgress();
    public final native boolean nativeCancel(long handle);
    public final native void nativeDestroy(long handle);
    public final native String nativeMemProbe(String jobJson);
    public final native String nativeVersions();
}
