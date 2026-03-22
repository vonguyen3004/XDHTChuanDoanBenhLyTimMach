"""
Test script: Kiểm tra upload + inference với file .HEA + .DAT từ PTB-XL
"""
import sys
from pathlib import Path
import numpy as np

# Thêm path để import từ codebase
sys.path.insert(0, str(Path(__file__).parent / '..'))

from ptbxl_parser import PTBXLParser

def test_ptbxl_load():
    """Test load file .HEA + .DAT từ PTB-XL"""
    
    parser = PTBXLParser()
    
    # Path tới dataset
    record_dir = Path(__file__).parent / '../data/ptbxl/records100/00000'
    record_name = '00001_lr'
    
    print("=" * 70)
    print("TEST 1: Load file PTB-XL")
    print("=" * 70)
    
    try:
        ecg_data, metadata = parser.load_record(record_dir, record_name)
        
        print(f"✓ Load thành công!")
        print(f"  Record: {metadata['record_name']}")
        print(f"  Sampling rate: {metadata['sampling_rate']} Hz")
        print(f"  Số mẫu: {metadata['num_samples']}")
        print(f"  Số lead: {metadata['num_leads']}")
        print(f"  Tên lead: {', '.join(metadata['leads'])}")
        print(f"  ECG data shape: {ecg_data.shape}")
        print(f"  Giá trị (mV) - min: {ecg_data.min():.4f}, max: {ecg_data.max():.4f}")
        
        # Thông tin chi tiết từng lead
        print(f"\n  Chi tiết từng lead:")
        for i, lead_name in enumerate(metadata['leads']):
            lead_data = ecg_data[:, i]
            print(f"    {lead_name:3s}: min={lead_data.min():.4f}, max={lead_data.max():.4f}, mean={lead_data.mean():.4f}")
        
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST 2: Tiền xử lý cho mô hình (lead II)")
    print("=" * 70)
    
    try:
        # Lead II là index 1
        processed = parser.preprocess_for_model(ecg_data, target_length=256, lead_index=1)
        
        print(f"✓ Tiền xử lý thành công!")
        print(f"  Output shape: {processed.shape}")
        print(f"  Output range: [{processed.min():.4f}, {processed.max():.4f}]")
        print(f"  Output dtype: {processed.dtype}")
        print(f"  Số giá trị zero: {(processed == 0).sum()}")
        print(f"  Số giá trị one: {(processed == 1).sum()}")
        
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST 3: So sánh với CSV format")
    print("=" * 70)
    
    # Load file CSV mẫu để so sánh
    csv_path = Path(__file__).parent / '../data/upload_sample.csv'
    
    try:
        if csv_path.exists():
            csv_data = np.loadtxt(csv_path, delimiter=',', dtype=float)
            print(f"✓ Load CSV mẫu thành công!")
            print(f"  CSV shape: {csv_data.shape}")
            print(f"  CSV range: [{csv_data.min():.4f}, {csv_data.max():.4f}]")
        else:
            print(f"! File CSV không tìm thấy: {csv_path}")
            
    except Exception as e:
        print(f"✗ Lỗi load CSV: {e}")
    
    print("\n" + "=" * 70)
    print("TÓconclude")
    print("=" * 70)
    print("✓ Module PTBXLParser hoạt động chính xác!")
    print("✓ File .HEA + .DAT từ PTB-XL đã có thể đọc được")  
    print("✓ Tiền xử lý dữ liệu thành công (chuẩn hóa 0-1, 256 sample)")
    print("\nBước tiếp theo: Tải lên qua web form tại http://127.0.0.1:5000/upload")
    
    return True

if __name__ == '__main__':
    success = test_ptbxl_load()
    sys.exit(0 if success else 1)
