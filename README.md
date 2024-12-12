#Speech to Text

![whisper-small](whisper-small.png)
![CUDA](cuda.png)
![pytorch](pytorch.png)

This is a pytorch application that uses the whisper model to transcribe audio files. In the transcribe.py script, we're using "whisper-small" from OpenAI's Whisper family, as shown in these specific lines:

```python
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

## Project Structure

Create the necessary directories:

```bash
mkdir input_audio output_text
```
## Usage Instructions

Place your WAV files in the input_audio directory

Run the script:

```bash
python transcribe.py
```
The transcribe.py script will:
1. Process each WAV file in the input directory
2. Create corresponding text files in the output directory
3. Utilize your GPU for faster processing
4. Handle different audio formats and sample rates
5. Convert stereo to mono if needed

## Performance Optimization

For your RTX 3050, add these lines at the start of the script for better performance:

```python
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
```

You can also adjust the model size based on your needs:
- whisper-tiny: Fastest but less accurate
- whisper-base: Good balance
- whisper-small: Better accuracy, still reasonable speed
- whisper-medium: More accurate but slower
- whisper-large: Most accurate but requires more GPU memory

This was tested on an ubuntu 24.04 system with 32GB RAM and RTX 3050 and was able to handle up to the medium model size comfortably.

