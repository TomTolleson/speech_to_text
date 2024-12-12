import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import torchaudio
from system_monitor import SystemMonitor

def check_cuda():
    print("\n=== CUDA Debug Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("No CUDA available - running on CPU")
        print("To use GPU, ensure NVIDIA drivers and CUDA toolkit are installed")
        return "cpu"

def setup_model(device):
    print("\n=== Model Setup ===")
    print(f"Loading Whisper model on device: {device}")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        forced_decoder_ids=None  # Disable default forced decoder IDs
    )
    
    if device == "cuda":
        try:
            model = model.to(device)
            print("Successfully moved model to GPU")
        except Exception as e:
            print(f"Error moving model to GPU: {e}")
            print("Falling back to CPU")
            device = "cpu"
    
    print(f"Model loaded successfully on {device}")
    return processor, model

def transcribe_audio(audio_path, processor, model, device):
    try:
        # Load and preprocess audio
        print(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        print(f"Original sample rate: {sample_rate}")
        if sample_rate != 16000:
            print("Resampling audio to 16kHz...")
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            print("Converting stereo to mono...")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process through model
        print("Processing through Whisper model...")
        inputs = processor(
            waveform[0].numpy(), 
            sampling_rate=16000, 
            return_tensors="pt",
            return_attention_mask=True
        )
        
        if device == "cuda":
            # Clear CUDA cache and sync before processing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            try:
                # Move inputs to GPU safely
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model = model.to(device)
                
                # Generate with reduced batch size and length
                print("Starting transcription generation...")
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    predicted_ids = model.generate(
                        inputs['input_features'],
                        attention_mask=inputs['attention_mask'],
                        max_length=224000,  # Reduced from 448000
                        num_beams=2,        # Reduced from 5
                        task='transcribe',
                        language='en'
                    )
                    torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"CUDA error: {e}")
                print("Falling back to CPU...")
                device = "cpu"
                model = model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                # Try again on CPU
                predicted_ids = model.generate(
                    inputs['input_features'],
                    attention_mask=inputs['attention_mask'],
                    max_length=224000,
                    num_beams=2,
                    task='transcribe',
                    language='en'
                )
        else:
            # CPU generation
            predicted_ids = model.generate(
                inputs['input_features'],
                attention_mask=inputs['attention_mask'],
                max_length=224000,
                num_beams=2,
                task='transcribe',
                language='en'
            )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        print(f"Supported formats: {torchaudio.list_audio_backends()}")
        raise

def main():
    # Initialize system monitor
    monitor = SystemMonitor()
    monitor.start()
    
    try:
        # Check CUDA availability and setup device
        device = check_cuda()
        
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        # Setup model
        processor, model = setup_model(device)
        
        # Process all WAV files in input directory
        input_dir = Path("input_audio")
        output_dir = Path("output_text")
        output_dir.mkdir(exist_ok=True)
        
        for audio_file in input_dir.glob("*.mp3"):
            print(f"Processing: {audio_file}")
            transcription = transcribe_audio(audio_file, processor, model, device)
            
            # Save to text file
            output_file = output_dir / f"{audio_file.stem}.txt"
            with open(output_file, "w") as f:
                f.write(transcription)
            print(f"Saved transcription to: {output_file}")
        
    finally:
        # Stop monitoring when done
        monitor.stop()

if __name__ == "__main__":
    main()


