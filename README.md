# Audio Recorder 錄音工具

A browser-based audio recorder that captures raw audio and Apple Dictation text simultaneously. Designed for use on iPad or MacBook, exporting files to a main Mac for Whisper + pyannote speaker diarization and LLM-assisted transcript correction.

專為 iPad 或 MacBook 設計的瀏覽器錄音工具，同步擷取原始錄音及 Apple 聽寫文字，匯出檔案至主 Mac 進行 Whisper + pyannote 說話者辨識及 LLM 轉錄校正。

---

## How It Works 運作方式

```
iPad / MacBook
  └── Audio Recorder (this page)
        ├── Records raw audio  →  2026-03-01_14-30-00.webm
        └── Captures Apple Dictation  →  2026-03-01_14-30-00_dictation.txt
              ↓
          Transfer to Main Mac
              ↓
          Whisper + pyannote  →  timestamped, speaker-labelled transcript
              ↓
          LLM + dictation.txt  →  corrected final transcript
```

---

## Recording Device Setup 錄音裝置設定

### Requirements 系統需求

No installation needed on the recording device. Just a modern browser.

錄音裝置無需安裝任何軟件，只需使用現代瀏覽器即可。

| Device | Recommended Browser |
|--------|-------------------|
| iPad | Safari or Chrome |
| MacBook | Chrome, Safari, or Firefox |

---

### Enabling Apple Dictation 啟用 Apple 聽寫

**Mac:**
1. System Settings → Keyboard → Dictation → Turn On
2. Press **Fn Fn** (or **Globe key** twice) to activate dictation
3. Make sure the dictation text box on the page is focused before speaking

**Mac：**
1. 系統設定 → 鍵盤 → 聽寫 → 開啟
2. 按兩下 **Fn**（或按兩下 **地球儀鍵**）啟動聽寫
3. 說話前請確保頁面上的聽寫文字框已被選取

**iPad:**
1. Settings → General → Keyboard → Enable Dictation → Turn On
2. Tap the microphone icon on the onscreen keyboard to activate
3. Tap the dictation text box on the page first, then activate dictation

**iPad：**
1. 設定 → 一般 → 鍵盤 → 啟用聽寫 → 開啟
2. 點按螢幕鍵盤上的麥克風圖示啟動聽寫
3. 先點按頁面上的聽寫文字框，再啟動聽寫

---

### Recording a Session 錄製步驟

1. Open **https://cfsthk.github.io/audio-recorder/** in your browser
2. Activate Apple Dictation and click into the **Apple Dictation** text box
3. Press **Record** — the waveform will show audio is being captured
4. Speak — Apple Dictation types into the text box automatically
5. Press **Stop** when done
6. Press **Export** — two files download automatically:
   - `YYYY-MM-DD_HH-MM-SS.webm` — raw audio
   - `YYYY-MM-DD_HH-MM-SS_dictation.txt` — dictation text
7. Transfer both files to your main Mac

---

1. 在瀏覽器開啟 **https://cfsthk.github.io/audio-recorder/**
2. 啟動 Apple 聽寫，點按頁面上的**聽寫文字框**
3. 按下 **Record** — 波形圖顯示正在錄音
4. 開始說話 — Apple 聽寫會自動將文字輸入文字框
5. 完成後按 **Stop**
6. 按 **Export** — 自動下載兩個檔案：
   - `YYYY-MM-DD_HH-MM-SS.webm` — 原始錄音
   - `YYYY-MM-DD_HH-MM-SS_dictation.txt` — 聽寫文字
7. 將兩個檔案傳送至主 Mac

---

## Main Mac Setup 主 Mac 設定

### Dependencies 依賴套件

Requires Python 3.9+. Install the following:

需要 Python 3.9 或以上版本，安裝以下套件：

```bash
pip install openai-whisper pyannote.audio torch torchaudio openai
```

**FFmpeg** (required by Whisper for audio decoding):

**FFmpeg**（Whisper 解碼音訊所需）：

```bash
# macOS (via Homebrew)
brew install ffmpeg
```

---

### Hugging Face Token (for pyannote) Hugging Face 存取金鑰

pyannote speaker diarization requires a free Hugging Face account and model access.

pyannote 說話者辨識需要免費的 Hugging Face 帳號及模型授權。

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the model terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Set it as an environment variable:

---

1. 在 [huggingface.co](https://huggingface.co) 建立免費帳號
2. 接受 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 的模型條款
3. 接受 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 的模型條款
4. 在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 產生讀取金鑰
5. 設定為環境變數：

```bash
export HF_TOKEN=hf_your_token_here
```

To make it permanent, add the line above to your `~/.zshrc` or `~/.bash_profile`.

如需永久設定，將上述指令加入 `~/.zshrc` 或 `~/.bash_profile`。

---

### Running Transcription 執行轉錄

Place your exported files in the same folder as `diarize_transcript.py`, then run:

將匯出的檔案放入與 `diarize_transcript.py` 相同的資料夾，然後執行：

```bash
python diarize_transcript.py --audio 2026-03-01_14-30-00.webm --dictation 2026-03-01_14-30-00_dictation.txt
```

**Output 輸出結果:**

```
[00:00 - 00:06]  SPEAKER_00:  你話飲檸檬茶嘅時候呀?
[00:06 - 00:12]  SPEAKER_01:  唔係檸檬水，因為飲咗檸檬茶。
...
```

The LLM uses the dictation text as a reference to correct proper nouns, names, and Cantonese-specific phrasing in the Whisper output.

LLM 會以聽寫文字作為參考，修正 Whisper 輸出中的人名、專有名詞及粵語特定用詞。

---

## File Naming 檔案命名

All files are named by recording timestamp automatically.

所有檔案均以錄音時間戳記自動命名。

| File | Description | 說明 |
|------|-------------|------|
| `YYYY-MM-DD_HH-MM-SS.webm` | Raw audio | 原始錄音 |
| `YYYY-MM-DD_HH-MM-SS_dictation.txt` | Apple Dictation text | Apple 聽寫文字 |

---

## Notes 注意事項

- **Browser mic permission:** The browser will ask for microphone access on first use. Allow it. / 瀏覽器首次使用時會要求麥克風權限，請允許。
- **Audio format:** Safari exports `.mp4`, Chrome/Firefox export `.webm`. Whisper handles both. / Safari 匯出 `.mp4`，Chrome/Firefox 匯出 `.webm`，Whisper 均支援。
- **Dictation accuracy:** Apple Dictation works best with Cantonese selected as the dictation language in system settings. / 在系統設定中選擇粵語作為聽寫語言，Apple 聽寫效果最佳。
- **First run:** pyannote downloads ~500MB of model weights on first use. Subsequent runs use the cached models. / pyannote 首次執行時會下載約 500MB 的模型，之後使用快取。