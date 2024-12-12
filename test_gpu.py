import torch
import sys

def test_gpu():
    print("\n=== GPU Test Results ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU computation
        print("\nRunning GPU computation test...")
        try:
            # Create test tensors
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            z = torch.matmul(x, y)
            
            print("✓ GPU computation test passed successfully!")
            print(f"Test tensor device: {z.device}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    else:
        print("❌ No CUDA device available")

if __name__ == "__main__":
    test_gpu() 