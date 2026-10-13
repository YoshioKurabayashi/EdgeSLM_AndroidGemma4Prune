package org.edgeslm.demo;

import java.io.File;

/** コンパイルの確かめ用（アプリには入れない）。実物は Kotlin の `class Gemma4Tokenizer`（companion object に load） */
public final class Gemma4Tokenizer {
    public static final Companion Companion = new Companion();
    private Gemma4Tokenizer() {}
    public final int[] encode(String text) { return new int[0]; }
    public static final class Companion {
        private Companion() {}
        public final Gemma4Tokenizer load(File dir) { return new Gemma4Tokenizer(); }
    }
}
